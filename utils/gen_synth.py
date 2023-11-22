from io import BytesIO
import json
import os
from PIL import Image
import arabic_reshaper
import fitz
from pdf2image import convert_from_path
import PyPDF2

from tqdm import tqdm
import pdfplumber


def generate_pdf(text, n_pages=1, font_size=10):
    doc = fitz.open()
    page = doc.new_page()
    text_reshaped = arabic_reshaper.reshape(text)
    text_writer = fitz.TextWriter(page.rect)
    text_writer.font = "Amiri"
    # text_writer.fontsize = font_size
    for i in range(0, 50):
        y0 = 50 + 15 * i
        if i == 0:
            text_writer.fill_textbox(
                (30, y0, 550, y0 + 30),
                text_reshaped,
                fontsize=int(20),
                align=3,
                right_to_left=True
            )
        elif i == 1:
            y0 = 50 + 30 * i
            text_writer.fill_textbox(
                (30, y0, 550, y0 + 15),
                text_reshaped,
                fontsize=int(font_size),
                align=3,
                right_to_left=True
            )
        else:
            text_writer.fill_textbox(
                (30, y0, 550, y0+15),
                text_reshaped,
                fontsize=int(font_size),
                align=3,
                right_to_left=True
            )
    ##############
    """text_reshaped0 = arabic_reshaper.reshape("تصميم المصاحف في تدوينة سابقة")
    text_writer = fitz.TextWriter(page.rect)
    text_writer.font = "Amiri"
    text_writer.fontsize = 80
    text_writer.fill_textbox(
        (500, 500, 550, 600),
        text_reshaped0,
        align=3,
        right_to_left=True
    )"""
    # image generation ( code not ready )
    text_writer.write_text(page)  # write text on each page
    # out = fitz.open()  # output PDF
    # making the pdf non-searchable
    """for page in doc:
        w, h = page.rect.br  # page width / height taken from bottom right point coords
        outpage = out.new_page(width=w, height=h)  # out page has same dimensions
        pix = page.get_pixmap(dpi=350)  # set desired resolution
        outpage.insert_image(page.rect, pixmap=pix)"""
    out_buffer = BytesIO()
    doc.save(out_buffer)  # out.save(out_buffer)
    out_buffer.seek(0)
    return out_buffer


def extract_text_and_coordinates_1(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in page.extract_text_lines():
                text = line["text"]
                x0, y0, x1, y1 = line["x0"], line["top"], line["x1"], line["bottom"]
                print(f"Text: {text}")
                print(f"Coordinates: (x0: {x0}, y0: {y0}), (x1: {x1}, y1: {y1})")
                print("---")


def extract_text_and_coordinates_onepage(pdf_path, image_path, output_dir="/home/omar/Documents"):
    image = Image.open(image_path)
    image_height, image_width = image.size[1], image.size[0]
    output_dict = {
        'text_line': [],
        'img_size': [image_height, image_width]
    }
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            for line in page.extract_text_lines():
                text = line["text"]
                # x0, y0, x1, y1 = line["x0"], line["top"], line["x1"], line["bottom"]
                x0, y0, x1, y1 = line["x0"] * (image_width / page.width), line["top"] * (image_height / page.height), \
                                 line["x1"] * (
                                         image_width / page.width), line["bottom"] * (image_height / page.height)
                print(f"Text: {text}")
                print(f"Coordinates: (x0: {x0}, y0: {y0}), (x1: {x1}, y1: {y1})")
                print("---")
                output_dict['text_line'].append({
                    'confidence': 1,
                    'polygon': [[round(x0), round(y0)], [round(x1), round(y0)],
                                [round(x1), round(y1)],
                                [round(x0), round(y1)]]
                })
    filename = image_path.split("/")[-1]
    output_filename = os.path.splitext(filename)[0] + '.json'
    # Write ground truth data to JSON files
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, 'w') as f:
        json.dump(output_dict, f, indent=4)


def extract_text_and_coordinates(pdf_path, output_dir="/home/omar/Documents"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in tqdm(enumerate(pdf.pages)):
            images = convert_from_path(pdf_path, first_page=i + 1, last_page=i + 1)
            for j, image in enumerate(images, start=1):
                image_path = f"{output_dir}/page_{i}.png"
                image.save(image_path)
                image = Image.open(image_path)
                image_height, image_width = image.size[1], image.size[0]
                output_dict = {
                    'text_line': [],
                    'img_size': [image_height, image_width]
                }
                for line in page.extract_text_lines():
                    text = line["text"]
                    # x0, y0, x1, y1 = line["x0"], line["top"], line["x1"], line["bottom"]
                    x0, y0, x1, y1 = line["x0"] * (image_width / page.width), line["top"] * (
                            image_height / page.height), line["x1"] * (
                                             image_width / page.width), line["bottom"] * (image_height / page.height)

                    output_dict['text_line'].append({
                        'confidence': 1,
                        'polygon': [[round(x0), round(y0)], [round(x1), round(y0)],
                                    [round(x1), round(y1)],
                                    [round(x0), round(y1)]]
                    })
                # filename = image_path.split("/")[-1]
                output_filename = f"page_{i + 1}.json"
                # Write ground truth data to JSON files
                output_path = os.path.join(output_dir, output_filename)
                with open(output_path, 'w') as f:
                    json.dump(output_dict, f, indent=4)


def extract_coordinates_pdf(pdf_path, output_dir="/home/omar/Documents"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for i in tqdm(range(len(pdf_reader.pages))):
            first_page = pdf_reader.pages[i]
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(first_page)

            path = f"{output_dir}/1ppdf"
            if not os.path.exists(output_dir):
                os.makedirs(f"{output_dir}/1ppdf")

            one_page_path = os.path.join(path, f"page{i}.pdf")

            with open(one_page_path, "wb") as output_file:
                pdf_writer.write(output_file)

            with pdfplumber.open(one_page_path) as pdf:
                for ii, page in enumerate(pdf.pages):
                    images = convert_from_path(one_page_path, first_page=ii, last_page=ii + 1)
                    for j, image in enumerate(images, start=1):
                        image_path = f"{output_dir}/page_{i}.png"
                        image.save(image_path)
                        image = Image.open(image_path)
                        image_height, image_width = image.size[1], image.size[0]
                        output_dict = {
                            'text_line': [],
                            'img_size': [image_height, image_width]
                        }
                        for line in page.extract_text_lines():
                            text = line["text"]
                            # x0, y0, x1, y1 = line["x0"], line["top"], line["x1"], line["bottom"]
                            x0, y0, x1, y1 = line["x0"] * (image_width / page.width), line["top"] * (
                                    image_height / page.height), line["x1"] * (
                                                     image_width / page.width), line["bottom"] * (
                                                     image_height / page.height)

                            output_dict['text_line'].append({
                                'confidence': 1,
                                'polygon': [[round(x0), round(y0)], [round(x1), round(y0)],
                                            [round(x1), round(y1)],
                                            [round(x0), round(y1)]]
                            })
                        # filename = image_path.split("/")[-1]
                        output_filename = f"page_{i}.json"
                        # Write ground truth data to JSON files
                        output_path = os.path.join(output_dir, output_filename)
                        with open(output_path, 'w') as f:
                            json.dump(output_dict, f, indent=4)


if __name__ == '__main__':
    text = "وكان أن أشرت الى آخر مشاريعي ألا وهو مصحف الإنترنت في معرض حديثي عن تصميم المصاحف في تدوينة سابقة، يسرني اليوم" \
           " ونحن نستقبل العشر المباركة من شهر ذي الحجة أن أعلن عن تشغيل موقع مصحف الإنترنت في نسخته التجريبية الأولى." \
           " وقد سمّيت المبادرة مشروع مصحف الإنترنت لما لها من الطابع التجريبي حيث لا يزال التطبيق في مرحلته الأولى"
    pdf_buffer = generate_pdf(text)
    # Save the PDF to a file
    output_path = "/home/omar/Videos/generate_syn/pymupdf/1.pdf"
    with open(output_path, "wb") as file:
        file.write(pdf_buffer.getvalue())
    # extract_text_and_coordinates_("/home/omar/Documents/Arabic-Quran.pdf", "/home/omar/Documents/image_Arabic-Quran_1.png")
    # extract_text_and_coordinates("/home/omar/Documents/journal1.pdf", "/home/omar/Documents/journal1")
    # extract_coordinates_pdf("/home/omar/Documents/journal1.pdf", "/home/omar/Documents/jj1")
    extract_coordinates_pdf("/home/omar/Documents/32067971.pdf", "/home/omar/Documents/jj6")
