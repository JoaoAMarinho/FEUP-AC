# Business Understanding

Banks are financial institutions, licensed to accept checking, saving deposits, and making loans. Regarding loans, it is the bank’s responsibility to accept or deny loan requests, based on the evaluation of the request. This is a decision of paramount importance that highly impacts the bank’s balance if wrongly taken. Successful loans will profit the bank, retrieving the applied fees. However, if unsuccessful, the bank might be at a loss of thousands of euros.

## Analysis of requirements with the end user

Given that the bank stores accurate data about their clients and previous records (transactions, granted loans, etc.), our product aims to help bank managers decide which loan requests should be accepted or not. This will provide help in the loan granting decision process, via a data-based input that will lead to better results, i.e. the bank loans will mostly succeed, resulting in a revenue increase by means of loan interest rates.

By automating the predicting process of loan success payment, our product will avoid money loss while improving decisions, and also decrease labor, thus reducing work expenses.

<p align="center" justify="center">
  <img src="./images/Data-Flow-Diagram.png"/>
</p>
<p align="center">
  <b><i>Fig 1. Data Flow Diagram</i></b>
</p>

## Definition of business goals

- Minimize the bank's bad credit by 4%
  - Minimize the number of unsuccessful loans to avoid money loss
  - Correctly predict, at least, 8 out of 10 loans
- Automate the process of deciding whether or not a loan should be granted
  - Decrease labor, aiming to decrease expenses
  - Accelerate the decision process
  - Unbias decisions
- The goal is achieved on time, i.e. the final model must be completed at the end of the due date

<!--
- Favour the denial of loans that will likely succeed other than accepting loans that will likely fail 

- What is our product for - provide the bank some knowledge and assurance about possible future loans
- This is done by implementing a model such that it is able to previously recognize loans that will not succeed
- End goal is to assure the bank has profit with loans and won't loose money - its all about da money
- In case of error, assure the model will not allow the bank to unsuccessfully grant a loan but prevent it from granting a successful loan (overall profit for the bank)

- Business goals help measure progress
- Business goals establish accountability
- Business goals improve decision-making
-->
## Translation of business goals into data mining goals

<!--
Why is AUC an appropriate metric for our problem?:

AUC provides an aggregate measure of performance across all possible classification thresholds.
AUC is scale-invariant. It measures how well predictions are ranked, rather than their absolute values.
AUC is classification-threshold-invariant. It measures the quality of the model’s predictions irrespective of what classification threshold is chosen.
-->
From a data mining perspective, our goal resides in predicting a target, the status of a loan, with the knowledge of other tables and features. 

- Building a model to predict the probability of a loan being successful
  - 1 if sucessful, -1 otherwise
- Obtain an AUC of, at least, 0.8
- Induce a higher weight for false positive outcomes, due to the greater impact they have in comparison to false negatives. Althought, aiming to decrease both results
