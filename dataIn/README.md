# Empirical Data for *"Emotion prediction as computation over a generative theory of mind"*

This readme describes the empirical data used by the Computed Appraisals Model. For context, see the [paper](https://daeh.info/pubs/houlihan2023computedappraisals.pdf) and main GitHub repository: [https://github.com/daeh/computed-appraisals](https://github.com/daeh/computed-appraisals).

Each subdirectory contains 4 files:

- `participant_data_composite.csv`
  - Participant-level data 
  - Each row corresponds to a single participant

- `participant_metadata_sidecar.csv`
  - Describes the participant-level data
  - Each row corresponds to a column in `participant_data_composite.csv`

- `trial_data_composite.csv`
  - Trial-level data
  - Each row corresponds to a participant's responses to a single trial

- `trial_metadata_sidecar.csv`
  - Describes the trial-level data
  - Each row corresponds to a column in `trial_data_composite.csv`


The behavioral paradigms used to collect the empirical data can be found on the [OSF repository](https://osf.io/yhwqn).

## Experiment key

| Data Set | Exp. |                      DV                       |                  IV                   |   Context   | $n$ pots | $n$ faces | $n$ trials: total (practice, analyzed) | $n$ participants collected | $n$ participants analyzed |
| :------: | :--: | :-------------------------------------------: | :-----------------------------------: | :---------: | :------: | :-------: | :------------------------------------: | :------------------------: | :-----------------------: |
|   (i)    |  6   |         $\omega^{base}, ~ \pi_{a_2}$          |     $a_1, ~ pot, ~ GenericPlayer$     | Anon. Game  |    44    |    12     |               12 (0, 12)               |            323             |            192            |
|   (ii)   |  9   | $\omega^{base}, ~ \omega^{repu}, ~ \pi_{a_2}$ |    $a_1, ~ pot, ~ SpecificPlayer$     | Public Game |    16    |    20     |                9 (1, 8)                |            413             |            181            |
|  (iii)   |  7   |                      $e$                      | $a_1, ~ a_2, ~ pot, ~ GenericPlayer$  | Public Game |    24    |     8     |                8 (0, 8)                |            607             |            339            |
|   (iv)   |  11  |                      $e$                      | $a_1, ~ a_2, ~ pot, ~ GenericPlayer$  | Public Game |    8     |     8     |                9 (1, 8)                |            391             |            215            |
|   (v)    |  10  |                      $e$                      | $a_1, ~ a_2, ~ pot, ~ SpecificPlayer$ | Public Game |    8     |    20     |                9 (1, 8)                |            2420            |           1512            |



Note that where attention/comprehension check columns appear in the `participant_data_composite.csv` files, blank values indicate a correct response (filled in values specify which incorrect response was made and `NR` indicates that a participant failed to respond to the question). When the value of  `randCondNum` is negative, it indicates that the webapp could not connect to the PHP server and served a default/random condition. 

## Data Set (i); Exp. 6

12 trials. Faces randomly paired with situations.

### Independent Variables (IV)

`faces = ['244_1', '250_1', '268_1', '272_1', '275_1', '276_1', '279_1', '283_1', '285_1', '286_1', '287_1', '288_1']`

`pots = [3.50, 30.00, 139.00, 269.50, 310.00, 822.50, 1030.50, 1116.50, 1283.50, 1562.50, 1588.00, 1598.50, 1803.00, 2300.50, 2318.00, 2835.50, 2843.50, 2983.00, 3331.00, 3532.50, 5322.50, 5958.00, 6559.00, 7726.50, 9675.00, 15744.50, 19025.50, 21719.50, 24145.00, 28802.00, 30304.50, 31403.00, 36159.00, 40606.00, 50221.00, 56488.00, 57518.00, 61381.50, 65673.50, 80954.50, 81899.00, 94819.00, 130884.00, 130944.00]`

### Dependent Variables (DV)

`q1responseArray`, `q2responseArray`, `q3responseArray` : "How much does this person value `<DYNAMIC>`?". Takes integer values from 0-48, anchored by "not at all", "a great deal".

`BTS_actual_otherDecisionConfidence` : "What did this person expect the other player to choose (and how confident is this person in their prediction about the other player's decision)?" Takes 6 point ordinal confidence values: `{ 0: "split, very confident", 1: "split, slightly confident", 2: "split, not confident", 3: "steal, not confident", 4: "steal, slightly confident", 5: "steal, very confident" }`.

### Attention/comprehension check

Participants were included in analyses if they correctly answered the 3 attention/comprehension questions in the `validationRadio` array and did not report prior exposure to the stimuli in the `val_recognized` free response. As noted in the `Notes` column: 1 participant did not fill out the comprehension check correctly but reported the correct answer in the feedback free responses, and was therefore included in the analysis; 1 participant reported not using the scales correctly and was excluded.

`validationRadio[0]`: "Which of these emotions is most similar to feeling contemptuous?"
--->`disdainful`

`validationRadio[1]`: "Imagine that the player shown above cares a great deal about sharing with the other player, not getting more than them. Is he more likely to choose Split or Steal? (You can ignore the other motivations that might factor into his decision.)"
--->`split` ("He probably Split.")

`validationRadio[2]`: "Which of these people is feeling the most joyful?"
--->`AF25HAS`



## Data Set (ii); Exp. 9

1 practice trial followed by 8 trials. 

### Independent Variables (IV)

`faces = ['239_1', '240_1', '246_1', '247_1', '249_2', '254_2', '255_2', '256_2', '263_1', '263_2', '266_2', '269_2', '270_1', '272_2', '276_1', '278_2', '280_1', '281_1', '285_1', '285_2']`

`pots = [124.00, 694.00, 1582.00, 5378.00, 10300.00, 12121.00, 19100.00, 27293.00, 33560.00, 48650.00, 61430.00, 67380.00, 84700.00, 138238.00, 140800.00, 162030.00]`

Practice trial always used: `{ "stimulus": "244_2", "pronoun": "she", "desc": "Customer service assistant", "pot": 1090, "decisionThis": "Stole" }`

See Exp. 10 for other IV.

### Dependent Variables (DV)

`q_bMoney_Array`, `q_bAIA_Array`, `q_bDIA_Array` : "How much does `<pronoun>` **actually** care about: `<DYNAMIC>`?". Takes integer values from 0-48, anchored by "not at all", "a great deal".

`q_rMoney_Array`, `q_rAIA_Array`, `q_rDIA_Array` : "How much does `<pronoun>` want a **reputation** for: `<DYNAMIC>`?". Takes integer values from 0-48, anchored by "not at all", "a great deal".

`BTS_actual_otherDecisionConfidence` : "What did this person expect the other player to choose (and how confident is this person in their prediction about the other player's decision)?" Takes 6 point ordinal confidence values: `{0:'split, very confident', 1:'split, slightly confident', 2:'split, not confident', 3:'steal, not confident', 4:'steal, slightly confident', 5:'steal, very confident'}`.

### Attention/comprehension check

Participants were included in analyses if they correctly answered the 5 attention/comprehension questions below and did not report prior exposure to the stimuli in the `val_recognized` free response. 

`val0(7510)`: "According to the host in this video, how large was this particular jackpot for these contestants? (You can replay the video if you'd like to hear the rules explained again.)"
--->`7510`

`val1(disdainful)`: "Which of these emotions is most similar to feeling contemptuous?"
--->`disdainful`

`val2(split)`: "Imagine that the player shown above cares a great deal about sharing with the other player, not getting more than them. Is he more likely to choose Split or Steal? (You can ignore the other motivations that might factor into his decision.)"
--->`split` ("He probably Split.")

`val3(three/rAIA)`: "Imagine that a player, Julia, has the values shown below. In the final round, Julia is facing off against her opponent, Roger. Which statement best describes Julia?"
--->`rAIA` (Julia really wants other people to believe that she cares about Roger's well-being more than she actually does.)

NB this question was not included for some participants. In these cases the response value is given as `-1`. The correct response was initially coded as `three` and was later changed to `rAIA`. Thus, this comprehension check evaluates to true for these three responses: `{-1, three, rAIA}`.

`val4(AF25HAS)`: "Which of these people is feeling the most joyful?"
--->`AF25HAS`



## Data Set (iii); Exp. 7

8 trials. Faces randomly paired with situations.

### Independent Variables (IV)

`faces = ['244_1', '250_1', '271_1', '272_1', '275_1', '283_1', '286_1',  '288_1']`

`pots = [2.00, 11.00, 25.00, 46.00, 77.00, 124.00, 194.00, 299.00, 457.00, 694.00, 1049.00, 1582.00, 2381.00, 3580.00, 5378.00, 8075.00, 12121.00, 18190.00, 27293.00, 40948.00, 61430.00, 92153.00, 138238.00, 207365.00]`

### Dependent Variables (DV)

`q1responseArray`, ... , `q20responseArray` : "How much `<DYNAMIC>` did this person feel?". Takes integer values from 0-48, anchored by "not any", "immense".

### Attention/comprehension check

Participants were included in analyses if they correctly answered the 5 attention/comprehension questions in the `validationRadio` array and did not report prior exposure to the stimuli in the `val_recognized` free response. As noted in the `Notes` column: 4 participants were excluded because they had prior knowledge of the Golden Balls gameshow; 2 participants reported not understanding the instructions and were excluded.

`validationRadio[0]`: "According to the host in this video, how large was this particular jackpot for these contestants? (You can replay the video if you'd like to hear the rules explained again.)"
--->`7510`

`validationRadio[1]`: "Which of these emotions is most similar to feeling contempt?"
--->`disdain`

`validationRadio[2]`: "Which of these emotions is most similar to feeling envy?"
--->`jealousy`

`validationRadio[3]`: "Which of these people is feeling the most joy?"
--->`AF25HAS`

`validationRadio[4]`: "Imagine that when this player learns the outcome of the game, she experiences the emotion shown on the slider (Devastation) at the specified level. What decision did the other player probably make?"
--->`steal` ("Other player probably Stole.")



## Data Set (iv); Exp. 11

1 practice trial followed by 8 trials. 

### Independent Variables (IV)

`faces = ['244_1', '250_1', '271_1', '272_1', '275_1', '283_1', '286_1',  '288_1']`

`pots = [124.00, 694.00, 1582.00, 5378.00, 12121.00, 27293.00, 61430.00, 138238.00]`

Practice trial always used: `{ "stimulus": "244_2", "pronoun": "she", "desc": "Customer service assistant", "pot": 1090, "decisionThis": "Stole", "decisionOther": "Split" }`

### Dependent Variables (DV)

`e_amusement`, ... , `e_sympathy` : "How much `<DYNAMIC>` did this person feel?". Takes integer values from 0-48, anchored by "not any", "immense".

### Attention/comprehension check

Participants were included in analyses if they correctly answered the 5 attention/comprehension questions below and did not report prior exposure to the stimuli in the `val_recognized` free response. 

`val0(7510)`: "According to the host in this video, how large was this particular jackpot for these contestants? (You can replay the video if you'd like to hear the rules explained again.)"
--->`7510`

`val1(disdain)`: "Which of these emotions is most similar to feeling contempt?"
--->`disdain`

`val2(jealousy)`: "Which of these emotions is most similar to feeling envy?"
--->`jealousy`

`val3(AF25HAS)`: "Which of these people is feeling the most joy?"
--->`AF25HAS`

`val4(steal)`: "Imagine that when this player learns the outcome of the game, she experiences the emotion shown on the slider (Devastation) at the specified level. What decision did the other player probably make?"
--->`steal` ("Other player probably Stole.")

#### Additional comprehension questions (not analyzed)

**These comprehension questions were collected but were not used as inclusion criteria and participants' responses and were not analyzed.**

`val5(pia2_D_a2_C)`: "Imagine a player, Roger, is in the final round with another player, Julia. Roger chose to Split. In which condition do you expect that Roger would feel the most Joy?"
--->`pia2_D_a2_C` ("He expected Julia to Steal but she Split.")

`val6(pia2_D_a2_C)`: "Imagine a player, Amanda, is in the final round with another player, Frank. Amanda chose to Split. In which condition do you expect that Amanda would feel the most Relief?"
--->`pia2_D_a2_C` ("She expected Frank to Steal but he Split.")



## Data Set (v); Exp. 10

1 practice trial followed by 8 trials. 

### Independent Variables (IV)

`faces = ['239_1', '240_1', '246_1', '247_1', '249_2', '254_2', '255_2', '256_2', '263_1', '263_2', '266_2', '269_2', '270_1', '272_2', '276_1', '278_2', '280_1', '281_1', '285_1', '285_2']`

`pots = [124.00, 694.00, 1582.00, 5378.00, 12121.00, 27293.00, 61430.00, 138238.00]`

Practice trial always used: `{ "stimulus": "244_2", "pronoun": "she", "desc": "Customer service assistant", "pot": 1090, "decisionThis": "Stole", "decisionOther": "Split" }`

```json
[
    { "stimulus": "272_2", "pronoun": "he", "desc": "Software engineer at Google" }, 
    { "stimulus": "278_2", "pronoun": "she", "desc": "Software engineer at Google" }, 

    { "stimulus": "254_2", "pronoun": "he",  "desc": "Janitor at an elementary school" }, 
    { "stimulus": "249_2", "pronoun": "she",  "desc": "Retired waitress" }, 

    { "stimulus": "240_1", "pronoun": "he",  "desc": "Pursuing a career in rap music" }, 
    { "stimulus": "285_2", "pronoun": "she",  "desc": "Design student, wants to be fashion designer" }, 

    { "stimulus": "281_1", "pronoun": "he",  "desc": "Executive of a petroleum energy company" }, 
    { "stimulus": "246_1", "pronoun": "she",  "desc": "Corporate lawyer" }, 

    { "stimulus": "255_2", "pronoun": "he",  "desc": "CEO of a global health non-profit" }, 
    { "stimulus": "276_1", "pronoun": "she",  "desc": "Doctor, volunteering in South Africa with 'Doctors Without Borders'" }, 

    { "stimulus": "285_1", "pronoun": "he", "desc": "Investment analyst at a hedge fund" }, 
    { "stimulus": "280_1", "pronoun": "she",  "desc": "Stock broker at an investment firm" }, 

    { "stimulus": "239_1", "pronoun": "he", "desc": "Hospital nurse" }, 
    { "stimulus": "270_1", "pronoun": "she",  "desc": "High school english teacher" }, 

    { "stimulus": "269_2", "pronoun": "he",  "desc": "Boxing coach" }, 
    { "stimulus": "256_2", "pronoun": "she",  "desc": "Swimming coach" }, 

    { "stimulus": "266_2", "pronoun": "he", "desc": "City council member, about to start campaigning for State Senate" }, 
    { "stimulus": "263_2", "pronoun": "she", "desc": "Child therapist and special education counselor" }, 

    { "stimulus": "263_1", "pronoun": "he", "desc": "Police officer" }, 
    { "stimulus": "247_1", "pronoun": "she", "desc": "Operates an elderly care center" }, 
]
```

### Dependent Variables (DV)

`e_amusement`, ... , `e_sympathy` : "How much `<DYNAMIC>` did this person feel?". Takes integer values from 0-48, anchored by "not any", "immense".

### Attention/comprehension check

Participants were included in analyses if they correctly answered the 5 attention/comprehension questions below and did not report prior exposure to the stimuli in the `val_recognized` free response. As noted in the `Notes` column: 1 participant reported not understanding the instructions and was excluded.

`val0(7510)`: "According to the host in this video, how large was this particular jackpot for these contestants? (You can replay the video if you'd like to hear the rules explained again.)"
--->`7510`

`val1(disdain)`: "Which of these emotions is most similar to feeling contempt?"
--->`disdain`

`val2(jealousy)`: "Which of these emotions is most similar to feeling envy?"
--->`jealousy`

`val3(AF25HAS)`: "Which of these people is feeling the most joy?"
--->`AF25HAS`

`val4(steal)`: "Imagine that when this player learns the outcome of the game, she experiences the emotion shown on the slider (Devastation) at the specified level. What decision did the other player probably make?"
--->`steal` ("Other player probably Stole.")

#### Additional comprehension questions (not analyzed)

**These comprehension questions were collected but were not used as inclusion criteria and participants' responses and were not analyzed.**

`val5(pia2_D_a2_C)`: "Imagine a player, Roger, is in the final round with another player, Julia. Roger chose to Split. In which condition do you expect that Roger would feel the most Joy?"
--->`pia2_D_a2_C` ("He expected Julia to Steal but she Split.")

`val6(pia2_D_a2_C)`: "Imagine a player, Amanda, is in the final round with another player, Frank. Amanda chose to Split. In which condition do you expect that Amanda would feel the most Relief?"
--->`pia2_D_a2_C` ("She expected Frank to Steal but he Split.")
