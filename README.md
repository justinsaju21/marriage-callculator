# 💍 Whom to Marry? - Candidate Ranking App

A Streamlit application that helps you make one of life's most important decisions using the **TOPSIS** (Technique for Order of Preference by Similarity to Ideal Solution) algorithm.

## 🌟 Features

-   **Scientific Ranking**: Uses the TOPSIS algorithm to rank candidates based on multiple criteria (Salary, Wealth, Height, etc.).
-   **Customizable Weights**: You decide what's important! Adjust the weight of each criteria using sliders.
-   **Dark Mode UI.**: A beautiful, premium glassmorphism design with a dark theme.
-   **Editable Data**: Add, remove, or modify potential candidates directly in the app.
-   **"Test Your Decisions"**: Pick a candidate you *think* is best, and see how they actually rank based on your set priorities.
-   **Data Persistence**: Your changes and weights are saved during your session.

## 🚀 How to Run

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/justinsaju21/marriage-callculator.git
    cd marriage-callculator
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**:
    ```bash
    streamlit run app.py
    ```

## 📂 Project Structure

-   `app.py`: Main application logic.
-   `utils/topsis.py`: Implementation of the TOPSIS ranking algorithm.
-   `data/`: Contains default CSV datasets for Grooms and Brides.
-   `style.css`: Custom CSS for the premium dark theme.

## 🛠️ Technologies Used

-   Python
-   Streamlit
-   Pandas
-   NumPy
