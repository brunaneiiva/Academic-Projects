
This folder contains two Microsoft Power Automate flows developed as part of my Master’s degree in Engineering. The flows aim to support the registration, approval, and monitoring of non-conformities within an organization, ensuring traceability and automated reporting.

The project is composed of two automated flows:

###  Flow 1 – Non-Conformities Registration and Approval 

 Flow 1 – Non-Conformities Registration and Approval

**Purpose:**  
Automates the process of registering a non-conformity (NC) via Microsoft Forms and managing its approval workflow.

**Main features:**
- Captures NC details from Microsoft Forms.
- Stores submissions in a SharePoint list (`T_NonConformities`).
- Sends email notifications to quality administrators.
- Triggers an approval process for the corrective action.
- If rejected, allows the administrator to suggest a new corrective action.
- Updates the SharePoint list with approval results and corrective action status.

**Technologies used:**  
Power Automate, Microsoft Forms, SharePoint, Outlook.

---

###  Flow 2 – Weekly Summary Report

**Purpose:**  
Sends a weekly summary of all non-conformities recorded in the previous 7 days.

**Main features:**
- Scheduled to run every week.
- Queries SharePoint items filtered by date.
- Dynamically builds an HTML table with NC details.
- Applies visual styles to the table for readability.
- Sends the summary by email to the quality manager.

**Technologies used:**  
Power Automate, SharePoint, Outlook, HTML (for table formatting).



