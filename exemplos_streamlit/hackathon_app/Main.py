import streamlit as st

st.set_page_config(
    page_title="Main",
    page_icon="??",
)

st.title("Interactive Visualization of Metaculus Forecasting Data")

st.header("Gapminder Visualization")

st.markdown(
    """
**Category:** Allows selection based on the top category of the category hierarchy of the question. When a question has more than one category it also considers the second category.

**Repeated Forecasts:** This metric shows the average number of forecasts adjustments done for each question. The higher the number the more times a user is adjusting the forecast.

**Accuracy Score:** For resolved questions, shows the average Log Score between prediction and resolution. This value is normalized to better align scores from binary and continuous questions.

**Deviation to MP:** Shows mean absolute error between prediction and Metaculus aggregated prediction.

**Deviation to CP:** Shows mean absolute error between prediction and community aggregated prediction.

**Number of Categories:** Shows number of different question categories user participated in. Log scale and capped at max 50.

**Number of Questions:** Shows number of different question user participated in. Log scale and capped at max 1500.

**Number of Forecasts:** Shows number of forecasts user did. Log scale and capped at max 3000.

**Reputation:** Shows user reputation level (quantile 0.9). Split into a level scale 0-5 where 5 is best reputation. Scale in Gapminder.py file.

**Seniority:** Shows user amount of active months spent in the platform. Split into a level scale 0-5 where 5 is largest number of active months. Scale in Gapminder.py file.

**Time scape:** The time scale is percentage value of the question duration before being resolved. For unresolved questions the duration is calculated with last forecast time in the dataset. For a more accurate view it is recommended to select only resolved questions.
"""
)

st.header("Heatmap Visualization")

st.markdown(
    """
**Number of months:** Selects only users where there were N months between first and last forecast.

**Activity+Reputation:** This metric combines the normalized reputation and normalized activity (1.5 x reputation + activity)

**Activity:** This metric shows normalized activity, measured as number of forecasts done for the month period. Log scale.

**Reputation:** This metric shows normalized max reputation for the user in the month period. Log scale.

**Questions:** This metric shows number of questions user did forecasts on in the month period.

"""
)
