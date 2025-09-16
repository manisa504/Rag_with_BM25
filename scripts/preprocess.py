"""
Pre-processes raw documents into a structured JSON format for ingestion.
"""
import os
import json
import argparse
import logging
from tqdm import tqdm
from rag.chunking import DocumentProcessor, Chunk

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def preprocess_documents(raw_dir: str, processed_dir: str):
    """
    Pre-processes all PDF documents in the raw directory and saves them as
    structured JSON files in the processed directory.
    """
    logger.info(f"Starting pre-processing from '{raw_dir}' to '{processed_dir}'")
    
    if not os.path.exists(raw_dir):
        logger.error(f"Raw directory not found: {raw_dir}")
        return

    os.makedirs(processed_dir, exist_ok=True)
    
    try:
        doc_processor = DocumentProcessor()
    except Exception as e:
        logger.error(f"Failed to initialize DocumentProcessor: {e}")
        logger.error("Please ensure all dependencies for 'docling' are installed in your environment.")
        return

    files_to_process = [f for f in os.listdir(raw_dir) if f.endswith(".pdf")]
    
    if not files_to_process:
        logger.warning(f"No PDF files found in {raw_dir}")
        return

    logger.info(f"Found {len(files_to_process)} PDF files to process.")

    for filename in tqdm(files_to_process, desc="Processing documents"):
        raw_path = os.path.join(raw_dir, filename)
        processed_path = os.path.join(processed_dir, f"{os.path.splitext(filename)[0]}.json")
        
        if os.path.exists(processed_path):
            logger.info(f"Skipping already processed file: {filename}")
            continue
            
        try:
            logger.info(f"Processing: {raw_path}")
            chunks = doc_processor.process(raw_path)
            
            if not chunks:
                logger.warning(f"No chunks were extracted from {filename}")
                continue

            # Convert chunks to a JSON-serializable format
            output_data = [chunk.to_dict() for chunk in chunks]
            
            with open(processed_path, 'w') as f:
                json.dump(output_data, f, indent=2)
                
            logger.info(f"Successfully processed and saved: {processed_path}")

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}", exc_info=True)

    logger.info("Document pre-processing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-process raw PDF documents for the RAG system.")
    parser.add_argument("--raw-dir", type=str, default="data/raw", help="Directory containing raw PDF files.")
    parser.add_argument("--processed-dir", type=str, default="data/processed", help="Directory to save processed JSON files.")
    args = parser.parse_args()
    
    preprocess_documents(args.raw_dir, args.processed_dir)
