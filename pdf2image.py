import tqdm
from PIL import Image
import fitz
import os

def pdf_to_images(pdf_dir, output_dir):

    pdf_file_list = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    pdf_file_list = [os.path.join(pdf_dir, f) for f in pdf_file_list]

    for pdf_file_path in pdf_file_list:
        print(f"Processing {pdf_file_path}")
        pdf_name = os.path.basename(pdf_file_path)
        pdf_id = os.path.splitext(pdf_name)[0]
        pdf_output_dir = os.path.join(output_dir, pdf_id)
        os.makedirs(pdf_output_dir, exist_ok=True)

        dpi = 200
        doc = fitz.open(pdf_file_path)

        for idx, page in enumerate(tqdm.tqdm(doc, desc=f"Processing {pdf_name}")):
            pix = page.get_pixmap(dpi=dpi)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            image_path = os.path.join(pdf_output_dir, f"{idx}.png")
            image.save(image_path)

if __name__ == "__main__":

    pdf_dir = "./benchmark/pdf/test"
    output_dir = "./benchmark/vlm/test"
    pdf_to_images(pdf_dir, output_dir)
