# Style transfer using a Linear Transformation

Here we have three pieces of audio. The first two are `Synth.wav` (audio $\mathbf{A}$) and `Piano.wav` (audio $\mathbf{B}$), which are recordings of a chromatic scale in a single octave played by a synthesizer and a piano respectively. The third piece of audio is the intro melody of “Blinding Lights” (audio $\mathbf{C}$) by The Weeknd, played with the same synth tone used to generate `Synth.wav`.

All audio files are in the `hw01_style_transfer/data` folder.

From these files, you can obtain the spectrogram $\mathbf{M}_A$, $\mathbf{M}_B$ and $\mathbf{M}_C$ . Your objective is to find the spectrogram of the piano version of the song “Blinding Lights” ($\mathbf{M}_D$).

In this problem, we assume that style can be transferred using a linear transformation. Formally, we need
to find the matrix $\mathbf{T}$ such that:

$$
\mathbf{TM}_A ≈ \mathbf{M}_B
$$

1. Write your code to determine matrix $\mathbf{T}$.

2. Our model assumes that $\mathbf{T}$ can transfer style from synthesizer music to piano music. Applying $\mathbf{T}$ on $\mathbf{M}_C$ should give us a estimation of “Blinding Lights” played by Piano, getting an estimation of $\mathbf{M}_D$. Using this matrix and phase matrix of $\mathbf{C}$, synthesize an audio signal.

    Submit your sythensized audio named as $\mathbf{problem3.wav}$.