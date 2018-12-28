import os
from os import walk
from fpdf import FPDF
import logging
import coloredlogs
from PyPDF2 import PdfFileMerger


logger = logging.getLogger()
coloredlogs.install(level='DEBUG')
coloredlogs.install(level='DEBUG', logger=logger)

class PdfCreator:
    def __init__(self, dir):
        self.files = [fname for (_, _, fname) in walk(dir)]
        self.images = [os.path.join(dir, fname) for fname in self.files[0] if 'png' in fname]
        self.pdfs = [os.path.join(dir, fname) for fname in self.files[0] if 'pdf' in fname]

    def create_pdf(self):
        pdf = FPDF()
        # imagelist is the list with all image filenames
        for idx, image in enumerate(self.images, start=1):
            logger.debug("file numer :{} from: {}".format(idx, len(self.images)))
            pdf.add_page()
            pdf.image(image)
        pdf.output("yourfile.pdf", "F")

    def merge_pdf(self):
        merger = PdfFileMerger()

        for pdf in self.pdfs:
            merger.append(open(pdf, 'rb'))

        with open('pdfmerged.pdf', 'wb') as fout:
            merger.write(fout)


pdf_creator = PdfCreator('../screenshots').create_pdf()
