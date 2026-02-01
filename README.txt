Dreamspace Audio Visulizer V1.0
abconnectorb
2026

This is a audio visualizer that aims to capture the asthetics of the dream background visuals of Pokemon Mystery Dungeon Explorers of Sky
 
Instructions:

1. Add your mp3 music file to the Dreamspace Audio Visualzer

2. Change the file name in main.py to your music file

3. SMILES GO FOR MILES! Sorry, I mean run main.py, then close the spectograph when it opens. You must have a frequency_bands.npy file after it completes to continue.

4. Configure render.py:
    change bpm to the bpm of your music file. This controls the crossing rate of the moving waves, fliping the percieved wave every beat.
    change colorSpeed to control the rate the background will cycle through colors. (0.5 default).
    change verticalScale to change how many waves will be displayed across the screen, lower values is more waves. Proportions will stay the same. PMD accurate is 
        about 0.8, but I recommend lower values.
    change loudnessInfluence to affect how much the wave will grow when subject to volume changes. 0 to 1 is usally ok, above 1 the waves may clip outside of the display 
        at high verticalScale values.
    near bottom of render.py find the commented DS RESOLUTION EFFECT and UPSCALE DS RESOLUTION EFFECT. Uncomment these if you wjsb to set the resolution to the DS.

5. Run render.py, this takes me about 5 min per minute of music to complete. 

6. Once completed, you will now find DreamspaceAV.mp4 in the project folder. That is the completed video, it wont have the audio attached.

scruube on Discord if you have any questions, happy to help. vibe coded