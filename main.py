import logging
import json
import re
import os
import spacy
import dateutil.parser
import pandas as pd
import time
from typing import Optional, Dict, Union, List
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from vertexai.generative_models import GenerativeModel
import vertexai
from playwright.sync_api import sync_playwright

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class AdvancedTextPreprocessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            self.logger.error("Spacy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')

        self.patterns = {
            'html': r'<[^>]+>',
            'emails': r'\S+@\S+',
            'urls': r'http\S+',
            'special_chars': r'[^\w\s.,!?;:()\'"/-]',
            'multiple_spaces': r'\s+',
            'bullet_points': r'[•●■◆▪️]+\s*',
            'multiple_newlines': r'\n+'
        }

    def preprocess(self, text: str, debug: bool = False) -> str:
        try:
            if debug:
                print("Original text:", text[:200] + "...")

            text = self._remove_patterns(text)
            text = self._standardize_dates(text)
            text = self._process_entities(text)
            text = self._normalize_whitespace(text)

            if debug:
                print("\nFinal processed text:", text[:200] + "...")

            return text.strip()
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            raise

    def _remove_patterns(self, text: str) -> str:
        for pattern_name, pattern in self.patterns.items():
            text = re.sub(pattern, ' ', text)
        return text

    def _standardize_dates(self, text: str) -> str:
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == 'DATE':
                try:
                    parsed_date = dateutil.parser.parse(ent.text)
                    standardized = parsed_date.strftime('%Y-%m-%d')
                    text = text.replace(ent.text, standardized)
                except:
                    continue
        return text

    def _process_entities(self, text: str) -> str:
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ['GPE', 'ORG']:
                text = text.replace(ent.text, ent.text.title())
        return text

    def _normalize_whitespace(self, text: str) -> str:
        return ' '.join(text.split())


class PDFProcessor:
    def __init__(
            self,
            project_id: str = "gcp-smart-capture",
            location: str = "us",
            processor_id: str = "6d88bf439f34e6a5"
    ):
        self.logger = logging.getLogger(__name__)
        try:
            self.client = documentai.DocumentProcessorServiceClient(
                client_options=ClientOptions(
                    api_endpoint=f"{location}-documentai.googleapis.com"
                )
            )
            self.resource_name = self.client.processor_path(
                project_id, location, processor_id
            )
        except Exception as e:
            self.logger.error(f"Failed to initialize PDFProcessor: {str(e)}")
            raise

    def process_pdf(self, file_path: str) -> str:
        try:
            self.logger.info(f"Processing PDF: {file_path}")
            with open(file_path, "rb") as pdf_file:
                content = pdf_file.read()

            raw_document = documentai.RawDocument(
                content=content,
                mime_type="application/pdf"
            )
            request = documentai.ProcessRequest(
                name=self.resource_name,
                raw_document=raw_document
            )
            result = self.client.process_document(request=request)
            self.logger.info("PDF processing completed successfully")
            return result.document.text
        except FileNotFoundError:
            self.logger.error(f"PDF file not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error processing PDF: {str(e)}")
            raise


class EventExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        try:
            vertexai.init(project="gcp-smart-capture", location="us-central1")
            self.model = GenerativeModel("gemini-1.5-pro-002")
            self.generation_config = {
                "candidate_count": 1,
                "max_output_tokens": 8192,
                "temperature": 0,
                "top_p": 0.95,
            }
        except Exception as e:
            self.logger.error(f"Failed to initialize EventExtractor: {str(e)}")
            raise

    def estimate_gemini_tokens(self, text: str) -> int:
        char_count = len(text)
        space_punct_count = len(re.findall(r'[\s\.,!?;:(){}\[\]"\'`~@#$%^&*\-+=|\\/<>]', text))
        estimated_tokens = (char_count // 4) + space_punct_count
        return estimated_tokens

    def extract_events(self, text: str) -> Union[Dict, str]:
        try:
            self.logger.info("Starting event extraction")
            input_token_count = self.estimate_gemini_tokens(text)

            schema = {
                "event_name": "",
                "location": "",
                "start_date": "",
                "end_date": "",
                "description": "",
                "event_type": "",
                "region": "",
                "owner": "",
                "event_url": "",
                "event_budget": 0.0
            }

            prompt = f"""
    You are a document entity extraction specialist. Given a document, extract these entities:
    {json.dumps([schema], indent=4)}

    Guidelines:
    - Follow the exact JSON schema shown above
    - Extract only text found in the document
    - Format dates as DD/MM/YYYY (assume 2024 if year not specified)
    - For date ranges: separate start_date and end_date
    - For event_type: use 'Online', 'In-Person', or 'On-Demand'
    - event_budget should be a float value (use 0.0 if not found)
    - For location: extract the specific venue or city
    - For region: extract the country or geographical region
    - For owner: extract the event organizer or responsible entity
    - Look for URLs in hyperlinks or text
    - Generate a 2-3 line description summarizing key details
    - Extract from both tabular and text formats
    - Use empty string "" for missing fields

    Text: {text}

    Return the events in the exact JSON format shown above.
    """
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config
            )

            response_text = response.text.strip()

            try:
                start_idx = response_text.find('[')
                end_idx = response_text.rfind(']')

                if start_idx != -1 and end_idx != -1:
                    json_str = response_text[start_idx:end_idx + 1]
                    events = json.loads(json_str)
                    return {
                        "events": events,
                        "input_token_count": input_token_count
                    }
                else:
                    self.logger.warning("No JSON structure found in response")
                    return {
                        "events": [],
                        "raw_response": response_text,
                        "input_token_count": input_token_count
                    }

            except json.JSONDecodeError as e:
                self.logger.warning(f"Failed to parse JSON: {str(e)}")
                return {
                    "events": [],
                    "raw_response": response_text,
                    "input_token_count": input_token_count
                }

        except Exception as e:
            self.logger.error(f"Error during event extraction: {str(e)}")
            return {
                "events": [],
                "error": str(e),
                "input_token_count": 0
            }


class EventJSONProcessor:
    def __init__(self, output_file: str = "extracted_events.json"):
        self.output_file = output_file
        self.event_occurrences = {}
        self.logger = logging.getLogger(__name__)

    def process_json_to_file(self, json_result: Dict, pdf_name: str, preprocessing_time: float,
                             total_time: float) -> None:
        try:
            events = json_result.get("events", [])
            input_token_count = json_result.get("input_token_count", 0)
            processed_events = []

            for event in events:
                event_key = f"{event.get('event_name', '')}-{event.get('location', '')}-{event.get('start_date', '')}"
                self.event_occurrences[event_key] = self.event_occurrences.get(event_key, 0) + 1

                processed_event = {
                    'event_name': event.get('event_name', ''),
                    'location': event.get('location', ''),
                    'start_date': event.get('start_date', ''),
                    'end_date': event.get('end_date', ''),
                    'description': event.get('description', ''),
                    'event_type': event.get('event_type', ''),
                    'region': event.get('region', ''),
                    'owner': event.get('owner', ''),
                    'event_url': event.get('event_url', ''),
                    'event_budget': event.get('event_budget', 0.0),
                    'metadata': {
                        'pdf_name': pdf_name,
                        'preprocessing_time': round(preprocessing_time, 3),
                        'total_processing_time': round(total_time, 3),
                        'token_count': input_token_count,
                        'occurrence_number': self.event_occurrences[event_key]
                    }
                }
                processed_events.append(processed_event)

            # Load existing data if file exists
            existing_data = []
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, 'r', encoding='utf-8') as f:
                        existing_data = json.load(f)
                except json.JSONDecodeError:
                    self.logger.warning(f"Could not parse existing JSON file: {self.output_file}")
                except Exception as e:
                    self.logger.error(f"Error reading existing JSON file: {str(e)}")

            # Combine existing data with new data
            all_events = existing_data + processed_events

            # Write back to file with pretty printing
            with open(self.output_file, 'w', encoding='utf-8') as f:
                json.dump(all_events, f, indent=4, ensure_ascii=False)

            self.logger.info(f"Successfully wrote {len(processed_events)} events to {self.output_file}")

        except Exception as e:
            self.logger.error(f"Error processing events to JSON: {str(e)}")
            raise


def save_pdf(url: str, output_file: str, timeout: int = 60000, retries: int = 3) -> bool:
    """
    Save a webpage as a PDF file using Playwright's sync API with enhanced error handling.
    """
    logger.info(f"Saving PDF for URL: {url}")

    for attempt in range(retries):
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page = browser.new_page()

                page.set_default_navigation_timeout(timeout)
                page.set_default_timeout(timeout)

                logger.info(f"Attempt {attempt + 1}/{retries}: Loading page")

                try:
                    page.goto(url, wait_until="networkidle", timeout=timeout)
                except Exception as e:
                    logger.warning(f"networkidle timeout, falling back to load: {str(e)}")
                    page.goto(url, wait_until="load", timeout=timeout)

                try:
                    page.wait_for_load_state("domcontentloaded", timeout=timeout)
                    time.sleep(2)
                except Exception as e:
                    logger.warning(f"Additional wait states failed: {str(e)}")

                logger.info("Generating PDF")
                page.pdf(path=output_file, format="A4", print_background=True)
                logger.info(f"PDF saved to {output_file}")

                browser.close()
                logger.info("Browser closed successfully")
                return True

        except Exception as e:
            logger.error(f"Attempt {attempt + 1}/{retries} failed: {str(e)}")
            if attempt == retries - 1:
                logger.error(f"All {retries} attempts failed for URL {url}")
                return False
            time.sleep(2 ** attempt)  # Exponential backoff

    return False


class TimedEnhancedDocumentProcessor:
    def __init__(self, debug_mode: bool = False):
        self.logger = logging.getLogger(__name__)
        self.pdf_processor = PDFProcessor()
        self.text_preprocessor = AdvancedTextPreprocessor()
        self.event_extractor = EventExtractor()
        self.json_processor = EventJSONProcessor()
        self.debug_mode = debug_mode
        self.default_timeout = 60000
        self.max_retries = 3

    def process_single_url(self, url: str, output_dir: str = "output_pdfs") -> Dict:
        start_total = time.time()
        try:
            os.makedirs(output_dir, exist_ok=True)

            pdf_filename = re.sub(r'[^\w]', '_', url) + '.pdf'
            pdf_path = os.path.join(output_dir, pdf_filename)

            self.logger.info(f"Starting to process URL: {url}")
            success = save_pdf(
                url,
                pdf_path,
                timeout=self.default_timeout,
                retries=self.max_retries
            )

            if not success:
                return {
                    "success": False,
                    "error": "Failed to save PDF from URL after multiple attempts",
                    "processing_time": time.time() - start_total,
                    "url": url
                }

            result = self.process_document(pdf_path)
            result['success'] = True
            result['pdf_path'] = pdf_path
            result['processing_time'] = time.time() - start_total
            result['url'] = url

            return result

        except Exception as e:
            self.logger.error(f"Error processing URL {url}: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_total,
                "url": url
            }

    def process_document(self, pdf_path: str) -> Dict:
        start_total = time.time()
        try:
            self.logger.info(f"Starting document processing for: {pdf_path}")

            raw_text = self.pdf_processor.process_pdf(pdf_path)

            start_preprocess = time.time()
            processed_text = self.text_preprocessor.preprocess(raw_text, debug=self.debug_mode)
            preprocessing_time = time.time() - start_preprocess

            result = self.event_extractor.extract_events(processed_text)

            total_time = time.time() - start_total

            pdf_name = os.path.basename(pdf_path)
            self.json_processor.process_json_to_file(result, pdf_name, preprocessing_time, total_time)

            result['preprocessing_time'] = preprocessing_time
            result['total_processing_time'] = total_time

            self.logger.info("Document processing completed successfully")
            return result
        except Exception as e:
            self.logger.error(f"Error during document processing: {str(e)}")
            return {"events": [], "error": str(e)}

def main():
    logging.info("Starting main application")
    try:
        # Example usage with a single URL
        url = "https://next2025challenge.devpost.com/"  # Replace with your actual URL
        processor = TimedEnhancedDocumentProcessor(debug_mode=True)

        # Configure logging for more detailed output
        logging.getLogger().setLevel(logging.DEBUG)

        result = processor.process_single_url(url)

        if not result['success']:
            print(f"\nProcessing failed for {result['url']}")
            print(f"Error: {result.get('error')}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")
            # You might want to implement fallback strategies here
        else:
            print(f"\nProcessing succeeded for {result['url']}")
            print(f"PDF saved at: {result['pdf_path']}")
            print(f"Processing time: {result['processing_time']:.2f} seconds")

            if result.get('events'):
                print("\nExtracted Events:")
                print(json.dumps(result['events'], indent=4, ensure_ascii=False))

    except Exception as e:
        logging.error(f"Application failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()