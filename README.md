# Instructions of launching the project via Jupyter notebook or script files
Steps to installing and launching the program.

## Prerequisites

1. Install Python (version 3.7 or later, project prepared on version 3.12).
2. Install `jupyter` using `pip` if it is not already installed:

   ```bash
   pip install notebook
   ```
3. Install required packages:
```bash
  pip install requirements.txt 
```
---

## Steps to Launch the program in Jupyter Notebook

1. Using CLI, navigate to the directory where the Spotify_YouTubeSongs.ipynb file is located.

2. Start Jupyter Notebook by typing:

   ```bash
   jupyter notebook
   ```

   This will open a new browser window showing the Jupyter Notebook interface.
3. In the Jupyter interface, navigate to the Spotify_YouTubeSongs.ipynb file and launch it.

---

## Running the Notebook

Once the notebook is open:

- To execute a cell, click on it and press `Shift + Enter` or click the "Run" button in the toolbar.
---

## Save the plots and summaries to a file

Run the [Optional] parts in the notebook or run the `data_analysis_main.py` file using Python in CLI environment to
automatically save all charts and summaries to disk. 

## Running the `data_analysis_main.py` file

1. Using CLI, navigate to SpotifySongs folder.
2. Run the script via:
```bash
  python data_analysis_main.py 
```
It will automatically save all plots and csv files to the disk. Behaviour is identical to running all [Optional] cells 
in the notebook version.