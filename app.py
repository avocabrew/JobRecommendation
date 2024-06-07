import streamlit as st
import pandas as pd
import numpy as np
import re
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import gdown
import nltk
import scipy

# Check scipy version
st.write(f"Scipy version: {scipy.__version__}")

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Mengunduh dataset dari Google Drive
url_dataset = 'https://drive.google.com/uc?id=10t5AQA0afmRD4Zl8gP_5WUXIZ5_C0NcR'
output_dataset = 'dataset.csv'
gdown.download(url_dataset, output_dataset, quiet=False)

# Mengunduh model dari Google Drive
url_model = 'https://drive.google.com/uc?id=1c7uj8FpTRU-xb_kIVWHv0wzDos34Fa-M'
output_model = 'model_components.pkl'
gdown.download(url_model, output_model, quiet=False)

# Load dataset
try:
    final_data = pd.read_csv(output_dataset)
except Exception as e:
    st.error(f"Error loading dataset: {e}")

try:
    final_data = final_data[final_data.state != 'No Info']
    final_data['state'] = final_data['state'].str.title()
    final_data['city'] = final_data['city'].str.title()
except Exception as e:
    st.error(f"Error processing dataset: {e}")

# Load precomputed embeddings and dataset if available (to save time)
try:
    with open(output_model, 'rb') as f:
        components = pickle.load(f)
    glove_vectors = components.get('glove_vectors')
    corpus_embeddings = components.get('corpus_embeddings')
except ImportError as e:
    st.error(f"Error loading model components: {e}")

# Check if corpus_embeddings is loaded correctly
if corpus_embeddings is None:
    st.error("corpus_embeddings is not loaded correctly.")

# Preprocessing tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
num_features = 300

def get_average_glove(tokens, model, num_features):
    valid_words = [word for word in tokens if word in model]
    if not valid_words:
        return np.zeros(num_features)
    return np.mean([model[word] for word in valid_words], axis=0)

# Streamlit app
st.title('Job Recommendation System US Based')

# Dropdown for category selection
try:
    state_option = final_data['state'].unique()
    state_option = np.sort(state_option)
    type_options = final_data['jenis_pekerjaan'].unique()
    sponsor_options = final_data['disponsori'].unique()
    apply_options = final_data['tipe_pendaftaran'].unique()

    state_option = np.concatenate([np.array(['Any']), state_option])
    type_options = np.concatenate([np.array(['Any']), type_options])
    sponsor_options = np.concatenate([np.array(['Any']), sponsor_options])
    apply_options = np.concatenate([np.array(['Any']), apply_options])

    selected_state = st.selectbox('Select State', state_option)
    if selected_state == 'Any':
        city_option = ['Any']
    else:
        city_option = final_data[final_data['state'] == selected_state]['city'].unique()
        city_option = np.concatenate([np.array(['Any']), city_option])

    city_option = np.sort(city_option)

    selected_city = st.selectbox('Select City', city_option)
    selected_type = st.selectbox('Select Job Type', type_options)
    selected_sponsor = st.selectbox('Select Sponsor Type', sponsor_options)
    selected_apply = st.selectbox('Select Application Type', apply_options)
except Exception as e:
    st.error(f"Error in dropdown selection: {e}")

city_filter = (final_data['city'] == selected_city) if selected_state != 'Any' else (final_data['city'] == final_data['city'])
type_filter = (final_data['jenis_pekerjaan'] == selected_type) if selected_type != 'Any' else (final_data['jenis_pekerjaan'] == final_data['jenis_pekerjaan'])
sponsor_filter = (final_data['disponsori'] == selected_sponsor) if selected_sponsor != 'Any' else (final_data['disponsori'] == final_data['disponsori'])
apply_filter = (final_data['tipe_pendaftaran'] == selected_apply) if selected_apply != 'Any' else (final_data['tipe_pendaftaran'] == final_data['tipe_pendaftaran'])

# Text box to input new skill description
new_text = st.text_area('Enter New Skill Description')

if st.button('Search Jobs'):
    if new_text:
        # Filter data based on selected categories
        filtered_data = final_data[(city_filter) & (type_filter) & (sponsor_filter) & (apply_filter)]
        
        if filtered_data.shape[0] < 3:
            st.write('Too Many Filters, Reduce Filters!')
        else:
            filtered_indices = filtered_data.index
            if corpus_embeddings is not None:
                filtered_embeddings = corpus_embeddings[filtered_indices]
            else:
                st.error("corpus_embeddings is not available.")
                filtered_embeddings = np.array([])

            # Preprocess and embed the new input text
            input_tokens = word_tokenize(new_text.lower())
            input_tokens = [lemmatizer.lemmatize(word) for word in input_tokens if word not in stop_words]
            input_embedding = get_average_glove(input_tokens, glove_vectors, num_features)

            if filtered_embeddings.size > 0:
                # Compute cosine similarities
                cosine_sim_new = cosine_similarity([input_embedding], filtered_embeddings).flatten()

                # Get top 10 similar jobs
                top_3_indices = cosine_sim_new.argsort()[-3:][::-1]
                top_3_titles = filtered_data.iloc[top_3_indices]['url_posting_pekerjaan']

                isBreak = False
                for i, index in enumerate(top_3_indices, 1):
                    if cosine_sim_new[index] < 0.6:
                        st.write('Your Skill Description is Too Short, Add More Details!')
                        isBreak = True
                        break

                if not isBreak:
                    st.write("Top 3 Jobs for You:")
                    for i, title in enumerate(top_3_titles, 1):
                        st.write(f"{i}. {title}.")
            else:
                st.write("No jobs available for the selected filters.")
    else:
        st.write("Enter a skill description to search for matching jobs.")
