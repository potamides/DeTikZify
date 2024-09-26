from collections import namedtuple
from functools import cache, cached_property
from io import BytesIO
from os import environ
from os.path import isfile, join
from re import MULTILINE, escape, findall, search
from subprocess import CalledProcessError, DEVNULL, TimeoutExpired
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Dict, Optional, Union

from PIL import Image
import pymupdf
from pdf2image.pdf2image import convert_from_bytes
from pdfCropMargins import crop
from transformers.utils import logging

from ..util import check_output, expand

logger = logging.get_logger("transformers")

class TikzDocument:
    """
    Facilitate some operations with TikZ code. To compile the images a full
    TeXLive installation is assumed to be on the PATH. Cropping additionally
    requires Ghostscript, and rasterization needs poppler.
    """
     # engines to try, could also try: https://tex.stackexchange.com/a/495999
    engines = ["pdflatex", "lualatex", "xelatex"]
    Output = namedtuple("Output", ['pdf', 'status', 'log'], defaults=[None, -1, ""])

    def __init__(self, code: str, timeout: Optional[int] = 60):
        self.code = code
        self.timeout = timeout

    @property
    def status(self) -> int:
        return self.compile().status

    @property
    def pdf(self) -> Optional[pymupdf.Document]: # type: ignore
        return self.compile().pdf

    @property
    def log(self) -> str:
        return self.compile().log

    @property
    def compiled_with_errors(self) -> bool:
        return self.status != 0

    @property
    def errors(self, rootfile: Optional[str] = None) -> Dict[int, str]:
        """
        Returns a dict of (linenr, errormsg) pairs. linenr==0 is a special
        value reserved for errors that do not have a linenumber in rootfile.
        """
        if self.compiled_with_errors:
            if not rootfile and (match:=search(r"^\((.+)$", self.log, MULTILINE)):
                rootfile = match.group(1)
            else:
                ValueError("rootfile not found!")

            errors = dict()
            for file, line, error in findall(r'^(.+):(\d+):(.+)$', self.log, MULTILINE):
                if file == rootfile:
                    errors[int(line)] = error.strip()
                else: # error occurred in other file
                    errors[0] = error.strip()

            return errors or {0: "Fatal error occurred, no output PDF file produced!"}
        return dict()

    @cached_property
    def is_rasterizable(self) -> bool:
        """true if we have an image"""
        return self.rasterize() is not None

    @cached_property
    def has_content(self) -> bool:
        """true if we have an image that isn't empty"""
        return (img:=self.rasterize()) is not None and img.getcolors(1) is None

    @classmethod
    def set_engines(cls, engines: Union[str, list]):
        cls.engines = [engines] if isinstance(engines, str) else engines

    @cache
    def compile(self) -> "Output":
        output = dict()
        with TemporaryDirectory() as tmpdirname:
            with NamedTemporaryFile(dir=tmpdirname, buffering=0) as tmpfile:
                codelines = self.code.split("\n")
                # make sure we don't have page numbers in compiled pdf (for cropping)
                codelines.insert(1, r"{cmd}\AtBeginDocument{{{cmd}}}".format(cmd=r"\thispagestyle{empty}\pagestyle{empty}"))
                tmpfile.write("\n".join(codelines).encode())

                try:
                    # compile
                    errorln, tmppdf, outpdf = -1, f"{tmpfile.name}.pdf", join(tmpdirname, "tikz.pdf")
                    open(f"{tmpfile.name}.bbl", 'a').close() # some classes expect a bibfile

                    def try_save_last_page():
                        try:
                            doc = pymupdf.open(tmppdf)
                            doc.select([len(doc)-1])
                            doc.save(outpdf)
                        except:
                            pass

                    for engine in self.engines:
                        try:
                            check_output(
                                cwd=tmpdirname,
                                timeout=self.timeout,
                                stderr=DEVNULL,
                                env=environ | dict(max_print_line="1000"), # improve formatting of log
                                args=["latexmk", "-f", "-nobibtex", "-norc", "-file-line-error", "-interaction=nonstopmode", f"-{engine}", tmpfile.name]
                            )
                        except (CalledProcessError, TimeoutExpired) as proc:
                            log = (getattr(proc, "output", b'') or b'').decode(errors="ignore")
                            error = search(rf'^{escape(tmpfile.name)}:(\d+):.+$', log, MULTILINE)
                            # only update status and log if first error occurs later than in previous engine
                            if (linenr:=int(error.group(1)) if error else 0) > errorln:
                                errorln = linenr
                                output.update(status=getattr(proc, 'returncode', -1), log=log)
                                try_save_last_page()
                        else:
                            output.update(status=0, log='')
                            try_save_last_page()
                            break

                    # crop
                    croppdf = f"{tmpfile.name}.crop"
                    crop(["-gsf", "-c", "gb", "-p", "0", "-a", "-1", "-o", croppdf, outpdf], quiet=True)
                    if isfile(croppdf):
                        output['pdf'] = pymupdf.open(croppdf)

                except FileNotFoundError:
                    logger.error("Missing dependencies: Did you install TeX Live?")
                except RuntimeError: # pdf error during cropping
                    pass

        if output.get("status") == 0 and not output.get("pdf", None):
            logger.warning("Could compile document but something seems to have gone wrong during cropping!")

        return self.Output(**output)

    def rasterize(self, size=420, expand_to_square=True) -> Optional[Image.Image]:
        if self.pdf:
            image = convert_from_bytes(self.pdf.tobytes(), size=size, single_file=True)[0]
            if expand_to_square:
                return expand(image, size)
            return image

    def save(self, filename: str, *args, **kwargs):
        match filename.split(".")[-1]:
            case "tex": content = self.code.encode()
            case "pdf" if self.pdf: content = self.pdf.tobytes()
            case fmt if img := self.rasterize(*args, **kwargs):
                img.save(imgByteArr:=BytesIO(), format=fmt)
                content = imgByteArr.getvalue()
            case fmt: raise ValueError(f"Couldn't save with format '{fmt}'!")

        with open(filename, "wb") as f:
            f.write(content)
