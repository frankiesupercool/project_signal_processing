# project_signal_processing

## Questions

- Length of the seminar paper presentation?
  - Answer: approx. 40min
- Presentation length of project?
  - Answer: No presentation - just lab report
- Length of lab report?
  - Answer: content conference paper but style basic report

## ssh sppc7 database access

- `ssh 0name@sppc7.informatik.uni-hamburg.de`

## Notes 

- preprocessing in dataset init function
- get item would be called everytime (each access, each epoch etc.)


- init should include preprocessing (Pytorch Lightning prepare_data for static one time operation)
- encoding both
- both modalities to handle in one dataset - else the audio would be loaded twice as the labels are coming from the audio

- cropping, noise, encoding not in dataset -> memory issue


- what we do: cropping separate/before from (noise? and) encoding, encoding in dataset:
- 1. Crop and create separate dataset (LRS3_cropped)
- 2. Load LRS3_cropped and apply everytime noise and different speakers and encoding
- 3. train