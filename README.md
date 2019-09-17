# Insight Proj

For the next 2 weeks of the INSIGHT program, this project aims to build a model that tracks the time-series motion of customers at a store from a surveillance camera without breaching customer privacy. I believe this is going to be a useful analysis tool for most business owners to learn potential insights about their business (i.e. store plan, pricing, etc). 

To achieve this, I plan to combine two different models: 

* person detection with a bounding box (i.e. SDD)
* bounding box tracking based on euclidian distance between bounding boxes or bounding box identification by one-shot-learning approach

In the end, the motion information of different customers will be visualized into a 2D heatmap. Depending on the specific business requirement: it could be averaged on a daily or weekly-basis || it could be categorized into different time throughout a day (morning/afternoon/night) so that the business owners can learn to optimize the usage of their store spaces by rearranging the product locations or putting discounts of the unpopular products, etc.

If extra time is allowed, a time-tracking feature will be added to identify the customers who need attention by tracking the lingering time of a customer at the same location.

The final model will be demonstrated on the actual surveillance camera footage examples.