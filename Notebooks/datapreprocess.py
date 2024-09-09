from bs4 import BeautifulSoup
import re
import pandas as pd
import html2text

# Initialize html2text converter


def remove_emojis(text):
    # Define a regex pattern to match emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)

def clean_text(text):
    if pd.isna(text):
        return ""
        
    # Remove any remaining HTML tags (as a fallback)
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ", strip=True)
    text = remove_emojis(text)
    # Remove special characters, URLs, and extra whitespace
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
    return text

def preprocess_data(df):
    # Clean text columns
    text_columns = ['product_title', 'product_description', 'product_bullet_point', 'product_brand']
    
    for col in text_columns:
        df[col] = df[col].apply(clean_text)
    
    # Merge relevant columns
    df['merged_text'] = df[['product_title', 'product_description', 'product_bullet_point', 
                            'product_brand', 'product_color', 'product_locale']].apply(
        lambda x: ' '.join(str(val) for val in x if pd.notna(val)), axis=1
    )
    
    df['merged_metadata'] = df[['product_title','product_brand', 'product_color', 'product_locale']].apply(
        lambda x: ' '.join(str(val) for val in x if pd.notna(val)), axis=1
    )


    # Create metadata column
    df['metadata'] = df.apply(lambda row: {
        'product_id': row['product_id'],
        'title': row['product_title'],
        'brand': row['product_brand'],
        'color': row['product_color'],
        'locale': row['product_locale']
    }, axis=1)

    # Select only relevant columns
    return df