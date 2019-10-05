# KNOWYOURMOVE

***KnowYourMove*** is a ComputerVision-based business intelligence tool that tracks the **motion** of individual customers from already-existing store surveillance camera systems. It analyzes the video and generates the information maps that can provide business owners with valuable insights to understand their customers better (in terms of their shopping paths and aggregated time spent in different areas). The model has been deployed as a web application [HERE](www.knowyourmove.store) where a user can upload a surveillance video to generate the reports. 

The information can be filtered by specific time windows (i.e. morning/afternoon or weekdays/weekends, etc). Such information is expected to provide time-specific customer traffics at a store so as to enable micro-optimization of business operation such as store plan and pricing strategy without breaching their privacy.

The model is based on:
* SSD (Single shot detector) model
* Centroid tracking algorithm
