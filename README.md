# arc-ai


Guide to Shapes

Filter by ID
- Returns: `1 x height x weight`

Length of Data
- **Train**: 400
- **Eval**: 400
- **Test**: 400


test_challenges_df, test_challenges_df, eval_challenges_df - All inputs and outputs for each dataset
- **Train**: List of dictionaries with
- - *Input*
- - *Output*
- **Test** List of dictionaries with
- - *Input*
- - *Output*

train_ids, test_ids, eval_ids - List of all ids per dataset (400, 100, 400)

train_set_df, test_set_df, eval_set_df - Training inputs and outputs per dataset (Num examples , training inputs , height , width)

train_predict_set_df, eval_predict_set_df - Testing inputs and solutions per dataset (Num examples , test input (1) , height , width)

