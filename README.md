# correction-detection
This project contains the code of *SAIF* (*Socially Aware personal assistant Implicit-Feedback correction detector*), a multimodal architecture which detects user corrections to an intelligent agent, taking as inputs the user’s voice commands as well as their transcripts.

The code consists of the files ```saif.py``` and ```emotions_model.sav```. The latter file is a pre-trained emotion recognition model from [Python Mini Project – Speech Emotion Recognition with librosa](https://data-flair.training/blogs/python-mini-project-speech-emotion-recognition/).

It also contains a labeled dataset of real interactions that users had while experimenting with the social agent *LIA* (*Learning by Instruction Agent*) \[[1](https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12383/12008),[2](https://www.tandfonline.com/doi/abs/10.1080/10447318.2018.1557972)\].
This dataset contains a series of 2540 pairs of spoken commands given to LIA by about 20 different users. For each command there is an original voice file and a written transcript produced by the ASR. Each command is followed by a response from the agent.
Each pair of consecutive commands is labeled according to whether the second one is a correction of the first.
There are three possible labels: no correction ("new command"), a correction in which the user provides a different command ("command correction"), and correction due to incorrect ASR transcription ("ASR correction").
There is also an indicator from LIA that specifies whether the command was executed successfully or not.

The dataset consists of the files ```transcriptVoiceMap.xlsx``` and ```wav.zip```.
