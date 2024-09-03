import streamlit as st
import random
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import plotly.graph_objects as go
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
import networkx as nx

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

# Advanced text generation using N-grams
class NGramModel:
    def __init__(self, n):
        self.n = n
        self.model = {}

    def train(self, text):
        words = word_tokenize(text.lower())
        for i in range(len(words) - self.n):
            gram = tuple(words[i:i+self.n])
            next_word = words[i+self.n]
            if gram not in self.model:
                self.model[gram] = {}
            if next_word not in self.model[gram]:
                self.model[gram][next_word] = 0
            self.model[gram][next_word] += 1

    def generate(self, length):
        current = random.choice(list(self.model.keys()))
        result = list(current)
        for _ in range(length - self.n):
            if current in self.model:
                next_word = random.choices(list(self.model[current].keys()),
                                           weights=list(self.model[current].values()))[0]
                result.append(next_word)
                current = tuple(result[-self.n:])
            else:
                break
        return ' '.join(result)

# Generate complex time series data
def generate_complex_time_series(days=365):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Trend component
    trend = np.linspace(0, 100, len(dates))
    
    # Seasonal component (yearly cycle)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(len(dates)) / 365)
    
    # Cyclical component (every 30 days)
    cyclical = 15 * np.sin(2 * np.pi * np.arange(len(dates)) / 30)
    
    # Random walk component
    random_walk = np.cumsum(np.random.randn(len(dates))) * 5
    
    # Combine components
    values = trend + seasonal + cyclical + random_walk
    
    return pd.Series(values, index=dates)

# Generate a color palette based on color theory
def generate_color_palette(base_color, n_colors):
    hue, saturation, value = rgb_to_hsv(*hex_to_rgb(base_color))
    palette = []
    for i in range(n_colors):
        new_hue = (hue + i * (360 / n_colors)) % 360
        palette.append(rgb_to_hex(*hsv_to_rgb(new_hue, saturation, value)))
    return palette

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/diff) + 120) % 360
    else:
        h = (60 * ((r-g)/diff) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (diff/mx) * 100
    v = mx * 100
    return h, s, v

def hsv_to_rgb(h, s, v):
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def main():
    st.title("Highly Advanced Procedurally Generated Streamlit App")

    # Sidebar for global controls
    st.sidebar.header("Global Controls")
    seed = st.sidebar.number_input("Random Seed", min_value=0, max_value=1000000, value=42)
    random.seed(seed)
    np.random.seed(seed)

    # Advanced text generation
    st.header("Advanced Text Generation and Analysis")
    ngram_model = NGramModel(3)
    ngram_model.train("This is a more advanced sample text to train our N-gram model for generating even more realistic random text that sounds coherent natural and contextually appropriate")
    text_length = st.slider("Choose text length", 20, 200, 100)
    generated_text = ngram_model.generate(text_length)
    st.write(generated_text)

    # Sentiment analysis
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(generated_text)
    st.write("Sentiment Analysis:")
    st.write(sentiment_scores)

    # Word cloud
    words = word_tokenize(generated_text.lower())
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    word_freq = pd.Series(filtered_words).value_counts()
    fig, ax = plt.subplots()
    word_freq[:20].plot(kind='bar')
    plt.title("Top 20 Words")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

    # Complex data generation and visualization
    st.header("Complex Data Generation and Visualization")
    n_series = st.slider("Number of data series", 1, 5, 3)
    data = {f"Series {i+1}": generate_complex_time_series() for i in range(n_series)}
    df = pd.DataFrame(data)

    # Interactive 3D visualization
    st.subheader("Interactive 3D Visualization")
    fig = go.Figure(data=[go.Surface(z=df.values.T)])
    fig.update_layout(title='3D Surface Plot of Time Series', autosize=False,
                      width=800, height=600,
                      margin=dict(l=65, r=50, b=65, t=90))
    st.plotly_chart(fig)

    # Machine Learning: K-means clustering
    st.subheader("K-means Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 5)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    df['Cluster'] = kmeans.fit_predict(scaled_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(n_clusters):
        cluster_data = df[df['Cluster'] == i]
        ax.scatter(cluster_data.index, cluster_data.iloc[:, 0], label=f'Cluster {i}')
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    plt.title("K-means Clustering of Time Series Data")
    st.pyplot(fig)

    # Network Graph
    st.header("Procedural Network Graph")
    n_nodes = st.slider("Number of nodes", 10, 50, 20)
    n_edges = st.slider("Number of edges", n_nodes-1, n_nodes*(n_nodes-1)//2, n_nodes)

    G = nx.gnm_random_graph(n_nodes, n_edges, seed=seed)
    pos = nx.spring_layout(G)

    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            reversescale=True,
            color=[],
            size=10,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'# of connections: {len(adjacencies[1])}')

    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Interactive Network Graph',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="",
                            showarrow=False,
                            xref="paper", yref="paper",
                            x=0.005, y=-0.002 ) ],
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    st.plotly_chart(fig)

    # Procedural art generation
    st.header("Advanced Procedural Art")
    art_type = st.radio("Choose art type", ["Fractal", "Particle System"])
    
    if art_type == "Fractal":
        # Mandelbrot set
        def mandelbrot(h, w, max_iter):
            y, x = np.ogrid[-1.4:1.4:h*1j, -2:0.8:w*1j]
            c = x + y*1j
            z = c
            divtime = max_iter + np.zeros(z.shape, dtype=int)
            for i in range(max_iter):
                z = z**2 + c
                diverge = z*np.conj(z) > 2**2
                div_now = diverge & (divtime == max_iter)
                divtime[div_now] = i
                z[diverge] = 2
            return divtime
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(mandelbrot(1000, 1000, 100), cmap='hot', extent=[-2, 0.8, -1.4, 1.4])
        ax.set_title("Mandelbrot Set")
        st.pyplot(fig)
    else:
        # Particle system
        n_particles = 1000
        positions = np.random.rand(n_particles, 2)
        velocities = np.random.randn(n_particles, 2) * 0.01
        
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(positions[:, 0], positions[:, 1], c=np.random.rand(n_particles), cmap='viridis', alpha=0.5)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title("Particle System")
        
        def update(frame):
            global positions, velocities
            positions += velocities
            positions = np.mod(positions, 1)  # Wrap around
            scatter.set_offsets(positions)
            return scatter,
        
        anim = animation.FuncAnimation(fig, update, frames=200, interval=50, blit=True)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
