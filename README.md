# Online Prediction System for Pregnancy-Related Disease Risks

This project is a web-based platform that implements a predictive model to assess the risk of 14 common pregnancy-related diseases. It is the deployed, operational counterpart to the models described in our research paper, providing tangible evidence of the system's real-world implementation.

## Live System Access

The live platform is deployed and publicly accessible at the following address:

**URL: [http://123.56.200.177/](http://123.56.200.177/)**

---

## Core Functionality

*   **Risk Probability Calculation:** The system accepts patient data via an Excel file and computes the risk probabilities for 14 distinct pregnancy-related diseases using our validated prediction models.

*   **Interactive Visualization:** Prediction results are presented through a clear and intuitive dashboard. Each of the 14 diseases has a dedicated circular pie chart that visually represents the calculated risk, with the exact probability percentage displayed at its center.

*   **Batch Processing:** The platform supports both single-patient predictions (via `.xlsx` upload) and batch predictions for multiple patients (via a `.zip`, `.rar`, or `.7z` archive containing multiple `.xlsx` files).

*   **Usage Analytics:** To demonstrate system-level deployment and engagement, the platform anonymously logs visit and prediction counts by geographical region based on the user's IP address.

---

## Instructions for Use

The system is designed for straightforward use by clinicians and researchers.

1.  **Download Template:** On the website, click the **"Download Template"** button to obtain the standardized `.xlsx` file for data entry.
2.  **Input Patient Data:** Populate the template with the patient's baseline and laboratory data according to the specified format.
3.  **Upload and Calculate:** Upload the completed file to the platform and click the **"Calculate"** button to initiate the risk assessment.
4.  **Review Results:** The prediction probabilities for all 14 diseases will be displayed on the results panel. For batch uploads, you can select individual patients from a dropdown menu to view their specific results.

---

## Technical Stack

*   **Backend:** FastAPI (Python), SQLAlchemy
*   **Frontend:** Vue.js, Element Plus, ECharts for data visualization
*   **Database:** SQLite for logging and statistics
