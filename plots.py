import matplotlib.pyplot as plt

def plot_name_trend(name, df):
    name = name.capitalize()  # Correct capitalization
    name_data = df[df['Name'] == name]
    name_data.sort_values(by="Year")
    
    # Create a figure and axis object
    fig, ax = plt.subplots()
    ax.plot(name_data['Year'], name_data['Count'], marker='o')
    ax.set_title(f'Popularity of the name {name} over time')
    ax.set_xlabel('Year')
    ax.set_ylabel('Count')
    
    return fig
