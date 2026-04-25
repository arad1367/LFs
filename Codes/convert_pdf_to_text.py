import os
import PyPDF2

def convert_pdfs_to_txt(input_folder="WhitePapers", output_folder="WhitePapers_TXT"):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all PDF files in the input folder
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.pdf')]
    print(f"Found {len(pdf_files)} PDF files to process.")

    for filename in pdf_files:
        pdf_path = os.path.join(input_folder, filename)
        txt_filename = filename.rsplit('.', 1)[0] + '.txt'
        txt_path = os.path.join(output_folder, txt_filename)

        try:
            with open(pdf_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                # Extract text from each page
                for page in reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"

            # Save the extracted text to a .txt file
            with open(txt_path, 'w', encoding='utf-8') as txt_file:
                txt_file.write(text)
                
            print(f"Successfully converted: {filename}")
            
        except Exception as e:
            print(f"Failed to convert {filename}. Error: {e}")

if __name__ == "__main__":
    convert_pdfs_to_txt()