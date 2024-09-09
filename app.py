import streamlit as st
import requests
import pandas as pd

st.set_page_config(page_title="Product Search Engine", layout="wide")

st.title("🔍 Advanced Product Search Engine")

# Center column for search
col1, col2, col3 = st.columns([1,2,1])

with col2:
    st.subheader("Search for Products")
    query = st.text_input("Enter your search query:", "dark blue french connection jeans for men")
    
    col_left, col_right = st.columns(2)
    with col_left:
        rerank_top_k = st.slider("Number of results:", 1, 20, 10)
    with col_right:
        alpha = st.slider("Alpha value:", 0.0, 1.0, 0.8, 0.1)
    
    if st.button("Search", type="primary", use_container_width=True):
        with st.spinner("Searching for products..."):
            response = requests.post("http://localhost:8000/search", 
                                     json={"query": query, "rerank_top_k": rerank_top_k, "alpha": alpha})
            
            if response.status_code == 200:
                results = response.json()
                df = pd.DataFrame(results)
                
                st.success(f"Found {len(results)} matching products!")
                
                # Display results in a table
                st.subheader("Search Results")
                st.dataframe(df, use_container_width=True)
                
                # Display product cards
                st.subheader("Top Products")
                for idx, product in enumerate(results[:6]):
                    st.markdown(f"""
                    <div style="border:1px solid #ddd; padding:10px; border-radius:5px; margin-bottom:10px;">
                        <h4>{product['title']}</h4>
                        <p><strong>Brand:</strong> {product['brand']}</p>
                        <p><strong>Color:</strong> {product['color']}</p>
                        <p><strong>Score:</strong> {product['scores']:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error("An error occurred while fetching results. Please try again.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and FastAPI")


# streamlit run app.py