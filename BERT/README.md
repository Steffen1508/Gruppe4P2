CHANGELOGS FOR BERT MODEL

V1: BERT implementeret i datasættet med BertForSequenceClassification som klassifikations model. Denne model finder ikke hver token, men laver binær klassifikation over hver span i datasættet, for at se om det indholder PII eller ej.

V2: BertForTokenClassification implementeret i stedet for, hvori 5 labels blev testet på over 3 epoch, 0,9 F1 score men på grund af stræk ubalance for datasættet, ville flere labels smadre performance.

V3: Modellen blev optimiseret til at vægte NO-PII med 0.1 fordi der er så stor ubalance, modellen testes med hele datasættet. 

V4: Hele datasættet bliver brugt til træning/validering/test nu frem for en nedcutted del, men det mistænkes at et for stort label map har været med til at forringe modellen. BERT trænes nu for at afgøre om det er et spørgsmål om at finde et sweetspot mellem for lidt labels (for lidt PII detektions muligheder) eller for mange labels (labels som bliver fejl klassificeret eller har for lidt data til at modellen med sikkerhed gætter rigtigt). Desuden er teorien at dette vil hjælpe på at hæve F1 score fra 0.89 til >0.90

BERT_inference: Dette script køre modellen "saved_model_reduced" på den oprettet PDF (alt her er 100% vibe coded) Men det viser hvordan den gemte model kan bruges i system kontekst så den ikke skal trænes hver gang. Det kræves at modellen downloaded fra "https://www.dropbox.com/t/yhU1bqLll8bMo166" da filen er for stor til at kunne pushes på GitHub