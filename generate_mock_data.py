import pandas as pd
import numpy as np

def generate_mock_data(filename="StudentsPerformance.csv"):
    np.random.seed(42)
    n = 1000
    data = {
        'gender': np.random.choice(['female', 'male'], n),
        'race/ethnicity': np.random.choice(['group A', 'group B', 'group C', 'group D', 'group E'], n),
        'parental level of education': np.random.choice(["bachelor's degree", 'some college', "master's degree", "associate's degree", 'high school', 'some high school'], n),
        'lunch': np.random.choice(['standard', 'free/reduced'], n),
        'test preparation course': np.random.choice(['none', 'completed'], n),
        'math score': np.random.randint(20, 101, n),
        'reading score': np.random.randint(20, 101, n),
        'writing score': np.random.randint(20, 101, n)
    }
    df = pd.DataFrame(data)
    
    # Adding a bit of correlation to make models perform better than random
    df['math score'] = (df['reading score'] * 0.4 + df['writing score'] * 0.4 + np.random.randint(0, 20, n)).clip(0, 100).astype(int)
    
    df.to_csv(filename, index=False)
    print(f"Mock dataset saved to {filename}")

if __name__ == "__main__":
    generate_mock_data()
