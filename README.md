# RecommendationSystem 
#### Bootcamp capstone project

A Recommendation system is a subclass of Information Filtering Systems which seek to predict a user's preference to an item. My team and I attempted to lay the foundation for improving user experience by suggesting music based on user ratings, user spotlights of music and music saves for an upcoming music sharing website.

### Exploratory Data Analysis of Various Music Features
* The below violin plot is to get a sense of how the music ratings were distributed. Members of the website can rate songs from 1 (a "miss") through 5 (a "hit").  The graph shows that more songs were rated a 5 than those on the lower end of the ratings spectrum, indicating that the ratings on the website tended to skew positive.
![Implicit Feedback](Images/dataviz1.png)

* On the Company's website, users can also save songs on their profile to listen to later. These saved songs are not visible to other users. The bar graph below shows the number of songs saved (y-axis) for each listener (x-axis).  It doesn't appear that many users have saved music on the website yet.
![Implicit Feedback](Images/dataviz2.png)

## Collaborative filtering
Collaborative filtering works on the priciple of collective intelligence - combining behavior, preferencs, or ideas of a group of people to derive insights. The assumption is that users who have similar preferences in the past are likely to have similar preferences in the future.
<img src="Images/collaborative1.jpg" width="300">

### Using Explicit Feedback
Explicit inout from the user regarding an item - prompting the user for rating. The accuracy of such a recommendation system depends on tge quantity of ratings provided by the user. It is seens as reliable data, provides transparency into the recommendation process. I used Matrix Factorization (Singular Value Decomposition).

#### Matrix Factorization
<img src="Images/MatrixFact.png" width="300">

### Using Implicit Feedback
The system will infer user's preferences by monitoring different user actions like links followed by a user, button clicks amongst others. The predictions are objective, as there is no bias. I identified music 'saved' and 'spotlighted' as our implicit features for the client. For deriving similarities I used Jaccard Distance. The formula compares abd produces a simple decimal statistic between 0 and 1.
![Implicit Feedback](Images/ImplicitFeedback.jpg)
#### Jaccard Index
![Jaccard Index](Images/toptal-blog-image-1423054884249.png)

## Hybrid Collaborative System
I implemented Weighted Hybrid Collaborative system to make recommendations. In this ratings of explicit and implicit recommendation techniques are combined together to produce a single recommendation.
![Jaccard Index](Images/toptal-blog-image-1423054884249.png)

## Flask App - RESTful API
![Flowchart](Images/Flowchart.png)


![Output](Images/Flask1.png)

## Improvements
* Once enough categorical data has been collected (ex: favorite genre, geographic location, birthday etc.), users can be filtered by category in accordance to some factors like song rating. 
* Artist followed data data can be treated as implicit data collection and used ti add weightage to recommendations. 


