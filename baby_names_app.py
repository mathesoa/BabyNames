import pandas as pd
import pickle
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from plots import plot_name_trend
from name_similarity import NameSimilarity
from generative_ai import load_model_and_tokenizer, generate_name


df = pd.read_csv('formatted_df_v2.csv')
highest_ranked_df = pd.read_csv('formatted_highest_rank.csv')
name_sim = NameSimilarity(db_path='baby_names.db')
combined_female_1 = pd.read_csv('combined_female_1.csv')
combined_female_2 = pd.read_csv('combined_female_2.csv')
combined_male = pd.read_csv('combined_male.csv')

st.title("Baby Name Animation")

st.write("""
The data presented here is based on baby name records from the [United States Social Security Administration (SSA)](https://www.ssa.gov/oact/babynames/), spanning the years 1880 to 2020. 
In this apps you can explore and visualize trends in baby names over time, including a bar chart race that shows the changing popularity of names year by year.
""")
# Upload the video of names by date
with st.container(border=True):
    option = st.selectbox("Baby Names from 1880 to 2020:", ("Girl", "Boy"))
    if option == "Girl":
        video_file = open('baby_girl_names_race_v01.mp4', 'rb')  # Open the .mp4 file in binary mode
    else: 
        video_file = open('baby_boy_names_race_v01.mp4', 'rb')
        
    video_bytes = video_file.read()         # Read the file into a byte stream
    st.video(video_bytes)

st.title("Baby Name Finder")
st.write("""
This section allows you to explore detailed statistics for individual baby names as well as discover new names based on your preferences. On the left, you can enter a specific name to view its historical trends, first recorded year, highest rank, and popularity. On the right, you can filter and find names based on the first letter, gender, or generation, and see popular, uncommon, and random name suggestions.
""")
with st.container(border=True):
# On one side, the user can
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Name Statistics")
        name = st.text_input("Look up a name:", "Mary").capitalize()
        st.write("The current name is:", name)

        if name in combined_female_1["Name"].values:
            female_names = combined_female_1[combined_female_1["Name"]==name]
            female_count = female_names["Count"].sum()
            selected_female_df = combined_female_1 
        elif name in combined_female_2["Name"].values:
            female_names = combined_female_2[combined_female_2["Name"]==name]
            female_count = female_names["Count"].sum()
            selected_female_df = combined_female_2 
        else:
            female_count = 0
            selected_female_df = combined_female_1 

        if name in combined_male["Name"].values: 
            male_names = combined_male[combined_male["Name"]==name]
            male_count = male_names["Count"].sum()
        else:
            male_count = 0

        if female_count >= male_count:
            name_stats_df = selected_female_df
        else:
            name_stats_df = combined_male

        if (female_count != 0) | (male_count != 0):
            st.pyplot(plot_name_trend(name, name_stats_df))
        
            name_df = name_stats_df[name_stats_df["Name"]==name]
            earliest_year = name_df["Year"].min()
            
            highest_rank = name_df["Rank"].min()
            
            highest_count = name_df["Count"].max()
            count_year_index = name_df["Count"].idxmax()
            highest_count_year = name_df["Year"].loc[count_year_index]
            
            st.write("This name was first recorded in:" , earliest_year)
            st.write("The highest rank for this name was:", highest_rank)
            st.write("The highest count for this name was:", highest_count, "in the year", highest_count_year)
        
            boy_or_girl = name_df["Gender"].mode()[0]
            if boy_or_girl == "M":
                st.write("This name is most commonly a baby boy name")
            else:
                st.write("This name is most commonly a baby girl name")
        else:
            st.write(f"The name {name} has not yet been recorded in the USA since the year 1880!")
    
    
    with col2:
        st.header("Find a Baby Name")
        letter = st.text_input("Choose the first letter", "A").capitalize()
        if len(letter) > 1:
            letter = letter[0]

        option = st.selectbox("Pick boy or girl:", ("Girl", "Boy"))
        generation = st.selectbox(
            "Selection a generation:", 
            ("Any", 
             "Lost Generation: 1883–1900", 
             "Greatest Generation: 1901–1927", 
             "Silent Generation: 1928–1945", 
             "Baby Boomers: 1946–1964", 
             "Generation X: 1965–1980", 
             "Millennials (Gen Y): 1981–1996", 
             "Generation Z: 1997–2012", 
             "Generation Alpha: 2013–present")
        )
    
        if option == "Girl":
            gender = "F"
        else:
            gender = "M"
    
        letter_df = df[(df["first_letter"] == letter) & (df["most_likely_gender"] == gender)]
    
        # Filter by generation if a specific generation is selected
        if generation != "Any":
            if "Lost Generation" in generation:
                start_year, end_year = 1879, 1900
            elif "Greatest Generation" in generation:
                start_year, end_year = 1901, 1927
            elif "Silent Generation" in generation:
                start_year, end_year = 1928, 1945
            elif "Baby Boomers" in generation:
                start_year, end_year = 1946, 1964
            elif "Generation X" in generation:
                start_year, end_year = 1965, 1980
            elif "Millennials" in generation:
                start_year, end_year = 1981, 1996
            elif "Generation Z" in generation:
                start_year, end_year = 1997, 2012
            elif "Generation Alpha" in generation:
                start_year, end_year = 2013, df["earliest_year"].max()  # Use max year for "present"
    
            # Apply the generation filter
            letter_df = letter_df[(letter_df["earliest_year"] >= start_year) & (letter_df["earliest_year"] <= end_year)]
    
        # Sort the data for highest and lowest ranking names
        letter_df_high = letter_df.sort_values(by=["highest_rank"], ascending=True)
        letter_df_low = letter_df.sort_values(by=["highest_rank"], ascending=False)
        
        highest_ranking_names = letter_df_high[["Name"]].head(10)
        lowest_ranking_names = letter_df_low[["Name"]].head(10)
        random_sample = letter_df[["Name"]].sample(n=10)
        
        st.write("Names that start with", letter, "in", generation)
        
        high_low_rank_df = pd.concat([highest_ranking_names.reset_index(drop=True), 
                             lowest_ranking_names.reset_index(drop=True),
                             random_sample.reset_index(drop=True)],
                             axis=1)
        high_low_rank_df.columns = ['Popular', 'Un-common', 'Random']
        st.write(high_low_rank_df)


st.title("Baby Name Similarity Finder Using NLP")
st.write("""
**Baby Name Similarity Finder Using NLP**: This tool uses **Natural Language Processing (NLP)** to find baby names similar to the one you enter. It applies **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to analyze names based on character patterns. By breaking down names into character-level n-grams, the tool calculates how closely related names are to the input.

Simply enter a name, and the system will return the most similar names, ranked by linguistic similarity. You can also visualize the results as a word cloud, where larger names indicate higher similarity.
""")
with st.container(border=True):
    
    input_name = st.text_input("Enter a name", "Anne").capitalize()
    
    # Input first letter from user (or you can make it dynamic)
    first_letter = input_name[0]
    
    num_names = st.slider("Select how many similar names to display", min_value=5, max_value=100, value=5, step=5)
    
    # Button to find similar names
    if st.button("Find Similar Names - Click for WordCloud"):
        if input_name:
            # Find similar names
            similar_names = name_sim.get_similar_names_for_letter(input_name, first_letter, top_n=num_names)
            
            if similar_names:
                st.write(f"Top {num_names} names similar to {input_name}:")
    
                # Prepare data for the word cloud (name: similarity score)
                word_freq = {name: score for name, score in similar_names}
    
                # Generate the word cloud
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    
                # Plot the word cloud using Matplotlib
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation="bilinear")
                plt.axis("off")
                plt.show()
    
                # Display the word cloud in Streamlit
                st.pyplot(plt)
            else:
                st.write(f"No similar names found for {input_name}.")
    
    
    # Get the top similar 1000 names into a DataFrame]
    similar_names_2 = name_sim.get_similar_names_for_letter(input_name, first_letter, top_n=1000)
    similar_names_df = pd.DataFrame(similar_names_2, columns=["Name", "Similarity"])
    
    # Merge with the highest_ranked_df to get generation data
    merged_df = pd.merge(similar_names_df, highest_ranked_df, on="Name", how="left")
    
    grouped_df = merged_df.groupby("Generation").head(10).sort_values(by=["Generation", "highest_rank"]).reset_index(drop=True)

    generation_dict = {}
    
    for index, row in grouped_df.iterrows():
        gen = row["Generation"]
        
        if gen in generation_dict:
            generation_dict[gen].append(row["Name"])
        else:
            generation_dict[gen] = [row["Name"]]
    generation_df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in generation_dict.items()]))
    
    # Display the DataFrame
    st.write("Names by Highest Ranking Generation")
    st.dataframe(generation_df)
    st.write("""
**How it works - detailed**

This tool uses advanced **Natural Language Processing (NLP)** techniques to find baby names similar to the one you enter. Specifically, it leverages **TF-IDF (Term Frequency-Inverse Document Frequency)** and **Cosine Similarity** to compute the similarity between names based on their character composition.

- **TF-IDF (Term Frequency-Inverse Document Frequency)**: This technique converts each name into a set of character-level n-grams (combinations of two or three consecutive characters). It then calculates the importance of each n-gram in distinguishing one name from others, helping capture subtle patterns between names like "Anne" and "Anna."
  
- **Cosine Similarity**: Once the names are transformed into TF-IDF vectors, Cosine Similarity measures how close the vectors are in multidimensional space. Names with higher similarity scores are more linguistically related to the input name.


1. You enter a name, and the system retrieves all names from the database starting with the same letter.
2. Using NLP, it computes the similarity between the input name and potential matches.
3. The system ranks the names by similarity and displays the top results.
4. You can also visualize the results as a word cloud, where the size of each name represents its similarity score.

For example, entering "Anne" might return similar names like "Anna," "Annie," or "Annabelle," ranked by similarity.
""")


st.title("Generative AI")
st.write("""
**Model Description**

This model uses a **Recurrent Neural Network (RNN)** with an **LSTM layer** to generate baby names based on character sequences. It works by learning patterns in names, predicting the next character from a given input.

- The **Embedding layer** converts characters into dense vectors.
- The **LSTM layer** captures the order and patterns in names.
- The final **Dense layer** with softmax outputs the next character based on probabilities.

The model is trained using 100 epochs to predict the next character in a sequence. You provide seed letters, and the model generates a new name based on its learned patterns.
""")

with st.container(border=True):

    model_type = st.selectbox(
    'Select Name Generation Model:',
    ('Gender Neutral', 'Boy', 'Girl')
    )

    # Input from user
    seed_text = st.text_input('Enter 1-4 seed letters:', 'J')
    if seed_text > 4:
        seed_text = seed_text[0:3]
    next_chars = st.slider('Select number of characters to generate:', 1, 10, 5)


# Load the selected model and tokenizer when the user presses the button
    if st.button('Generate Name'):
        model, tokenizer = load_model_and_tokenizer(model_type)
        generated_name = generate_name(model, tokenizer, seed_text, next_chars)
        st.write(f'Generated Name: {generated_name}')

