import fitz
from pathlib import Path

# marker = "UIN: IRDAN115RP0034V01201819"
PDF_DIR = Path("policy_pdfs")
INPUT_DIR = Path("input")
INPUT_DIR.mkdir(exist_ok=True)

marker = "customersupport@icicilombard.com"
stop_words = [
    # "policy ombudsman",
    "details of policy ombudsman",
    "details of insurance ombudsman are available below:",
    "customer information sheet",
]
if __name__ == "__main__":
    for pdf_file in PDF_DIR.glob("*.pdf"):
        print(f"Processing: {pdf_file.name}")
        # page = 1
        doc = fitz.open(pdf_file)
        policy_text = ""

        for page_num, page in enumerate(doc):
            # policy_text = ""

            page_text = page.get_text()
            #     # Initial check
            left_initial = fitz.Rect(0, 0, page.rect.width, page.rect.height)
            left_initial_text = page.get_text("text", clip=left_initial, sort=True)
            #     # print(f"Left initial text: {left_initial_text}")
            if any(s in left_initial_text.lower() for s in stop_words):
                print(f"Stopping at page {page_num}")
                break
            rect = page.rect

            width = rect.width
            height = rect.height

            #     # Adjust after testing
            if "personal-" in doc.name:
                header_height = 100  # page 1 only
                footer_height = 90  # all pages
                body_top = header_height
                if page_num == 22:
                    print(f"Stopping at page 22")
                    break
            else:
                header_height = 120
                footer_height = 90
                body_top = 0

                if page_num == 0:
                    body_top = header_height
                else:
                    body_top = 0

            # print(f"Page {page_num}: body_top = {body_top}")

            body_bottom = height - footer_height

            left_col = fitz.Rect(0, body_top, width / 2, body_bottom)

            right_col = fitz.Rect(width / 2, body_top, width, body_bottom)

            left_text = page.get_text("text", clip=left_col, sort=True)

            right_text = page.get_text("text", clip=right_col, sort=True)

            #     # if "policy ombudsman" in text.lower():
            page_text = left_text + "\n" + right_text + "\n"
            #     # all_text.append(page_text)

            #     # if page.number == 4:
            policy_text += page_text
            #     # break

        with open(f"input/{doc.name[12:-4]}.txt", "w") as f:
            # f.write(left_text)
            # f.write(right_text)
            # f.write("\n" + "=" * 80 + "\n")
            f.write(policy_text)
