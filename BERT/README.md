CHANGELOGS FOR BERT MODEL

V1: BERT implementeret i datasættet med BertForSequenceClassification som klassifikations model. Denne model finder ikke hver token, men laver binær klassifikation over hver span i datasættet, for at se om det indholder PII eller ej.

V2: BertForTokenClassification implementeret i stedet for, hvori 5 labels blev testet på over 3 epoch, 0,9 F1 score men på grund af stræk ubalance for datasættet, ville flere labels smadre performance.

V3: Modellen blev optimiseret til at vægte NO-PII med 0.1 fordi der er så stor ubalance, modellen testes med hele datasættet. 