from translation_engine import TranslationEngine
import sys

def main():
    print("\nEnglish to German Neural Machine Translation")
    print("=" * 50)
    print("Using Custom Transformer Architecture")
    print("-" * 50)
    
    # Initialize translation engine
    engine = TranslationEngine()
    
    while True:
        try:
            # Get input from user
            text = input("\nEnter English text (or 'q' to quit): ").strip()
            
            if text.lower() == 'q':
                print("\nThank you for using the translator!")
                break
            
            if not text:
                print("Please enter some text to translate.")
                continue
                
            # Translate
            translation = engine.translate(text)
            print(f"\nTranslation: {translation}")
            
        except KeyboardInterrupt:
            print("\n\nTranslation interrupted. Exiting...")
            sys.exit(0)
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
