import streamlit as st
import pandas as pd
import altair as alt
import pickle
from datetime import datetime
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter
from statistics import median

# VADER Setup
@st.cache
def download_vader():
    nltk.download("vader_lexicon")

download_vader()

# Increase the default width of Streamlit
st.markdown(f"""
            <style>
            .reportview-container .main .block-container{{
                max-width: 750px;
            }}
            </style>
            """,
            unsafe_allow_html=True)

st.write("""
    # Analyzing `/r/gunners` comments

    How do the users from the subreddit [/r/gunners](http://www.reddit.com/r/gunners)
    react after Arsenal wins and defeats?

    [by /u/cuchoi](www.fernandoi.cl)

    ### Motivation

    For me, *Post Match Threads* have always felt a bit over reactive: We win and we
    suddenly become a great team, Arteta becomes a genius, and our players are world
    class. We lose and Arteta is a donkey, Xhaka and Ceballos are liabilities, and we
    should sell all our players bar Tierny.

    I wanted to know if this "over reactiveness" could be seen in the sentiment of
    the comments. Also, I wanted to know if there are redditors that only comment
    when we lose and other that only comment when we win. *Post Match Threads*
    definetely feel like different places depending on the outcome of the game.

    ### The Data

    I scraped all the top level comments from each **"Post Match Thread"** from
    `/r/gunners` this season.

    I collected more than *8000 top-level comments from this season Post Match Threads*
    and calculated each comment their sentiment (a measure of how positive
    or negative the text is) using [VADER](https://github.com/cjhutto/vaderSentiment).

    A comment with -100 is a very negative comment, a zero is a neutral comment and 100
    is a very positive comment.

    The sentiment score is not perfect to measure overreactivenes towards our team.
    For example, let's compare two of the most negative comments from our win against Spurs in March:

    1. `Not an Arsenal fan, but wanted to come here and congratulate you on beating Spurs.
    Also fuck Lamela, dirty twat got what he deserved.`

    2. `what the fuck was that last 15. stupid mistakes, big block by Gabriel and me
    waiting for the customary Kane goal which did not come thankfully.`

    Both of these comments were labeled as extremely negative by VADER, but the first
    one is negative towards Spurs and the second one is negative towards our team.

    Ideally we would train a model to detect this difference but I won't do that today:
    In the analysis below I will take the average at the match or user level and use that
    as a proxy of the overall fan base sentiment.

    """)

@st.cache
def build_comments_df():
    """Use the downloaded dataset to build a comments dataframe"""
    DATASET_PATH = "dataset.pickle"
    with open(DATASET_PATH, 'rb') as handle:
        dataset_dict = pickle.load(handle)
    count = 0
    comment_records = []
    for post in dataset_dict:
        if "Post" not in post.title or 'Premier' not in post.title and post.id != "ircjx0":
            continue
        if post.created_utc < 1599000000:
            # Skip games before the first game of the 2020/2021 season (Arsenal 3 - Fullham 0)
            continue
        if post.id == "ircjxt":
            # Skip the Fulham game created by /u/GunnersMatchBot. Another thread was used.
            continue
        else:
            count += 1

            post_info = {'post_id':post.id,
                        'post_title': post.title,
                        'post_created_utc': int(post.created_utc),
                        'post': post}
            score_str = post.title.split(": ")[1].split("[English")[0]
            team1, team2 = score_str.split("-")[0], score_str.split("-")[1]
            score = lambda team_str: int(''.join(filter(str.isdigit, team_str)))
            if "Arsenal" in team1:
                arsenal_score = score(team1)
                other_score = score(team2)
            else:
                arsenal_score = score(team2)
                other_score = score(team1)

            post_info["arsenal_score"] = arsenal_score
            post_info["other_score"] = other_score

            # Get Match Result
            match_result = "tie"
            if arsenal_score > other_score:
                match_result = "won"
            elif arsenal_score < other_score:
                match_result = "lost"
            post_info["match_result"] = match_result


            for comment in dataset_dict[post]:
                record = {**post_info,
                        'id': comment.id,
                        'body': comment.body,
                        'author': comment.author,
                        'author_str': str(comment.author),
                        'created_utc': int(comment.created_utc),
                        'match_date': datetime.fromtimestamp(comment.created_utc),
                        'parent_id': comment.parent_id,
                        'permalink': comment.permalink,
                        'score': comment.score,
                        'comment_obj': comment}
                comment_records.append(record)

    df = pd.DataFrame(comment_records)
    df = df[df.id != 'g96zcpz'] # We remove this duplicated Man City post with 6 comments
    df.post_title = df.post_title.str.replace("Manchester City 0", "Manchester City 1") # Fix score issues

    sia = SentimentIntensityAnalyzer()
    vs = [sia.polarity_scores(comment)['compound'] for comment in df["body"]]
    df["sentiment"] = round(100*df["body"].apply(lambda body: sia.polarity_scores(body)['compound']),1)
    df['title_label'] = df['post_title'].str.split(":").str[1].str.split(" \[").str[0]
    return df

@st.cache
def build_match_sentiment_df(original_df):
    """Aggregate the comments at the match level and calculated match aggregations"""
    df = original_df.copy(deep=True)

    MAX_COM_LEN = 250
    most_positive_comment_idx = df[df.body.str.len()<MAX_COM_LEN].groupby(['post_title'])['sentiment'].idxmax()
    most_negative_comment_idx = df[df.body.str.len()<MAX_COM_LEN].groupby(['post_title'])['sentiment'].idxmin()
    def most_positive_comment(df):
        return df[df.index.isin(most_positive_comment_idx)]
    def most_negative_comment(df):
        return df[df.index.isin(most_negative_comment_idx)]

    match_sentiment = df.groupby('post_title').agg(sentiment=('sentiment', 'mean'),
                                                    match_date=('match_date', 'first'),
                                                    arsenal_score=('arsenal_score', 'first'),
                                                    other_score=('other_score', 'first'),
                                                    post_title=('post_title', 'first'),
                                                    title_label=('title_label', 'first'),
                                                    match_result=('match_result', 'first'),
                                                    created_utc=('created_utc', 'first'),
                                                    comment_num=('created_utc','count'),
                                                    post_id=('post_id', 'first'),
                                                    most_positive_comment=('body', most_positive_comment),
                                                    most_negative_comment=('body', most_negative_comment),
                                                    comment_count=('body', 'count'))
    match_sentiment['sentiment'] = round(match_sentiment['sentiment'])
    match_sentiment['match_date_title'] = match_sentiment["match_date"].dt.strftime("%Y-%m-%d") + " - " + match_sentiment['title_label']

    return match_sentiment

comments_df = build_comments_df()
matches_df = build_match_sentiment_df(comments_df)

st.write("""
         ## Findings

         ### Finding 1: Unsurprisingly, we tend to be more positive when we win

         Below you will see the average sentiment score for each match for the 2020-2021 season.

         As expected, when we win the average sentiment score is better and when we lose
         the sentiment score is lower. *Surpringsingly, our lowest is score is not
         a loss... it was the 1-1 tie against Burnley* (a very frustrating game indeed...).

         #### Average comment sentiment for each Premier League game
         """)
st.text("")
st.altair_chart(alt.Chart(matches_df).mark_bar().encode(
            x=alt.X('sentiment', axis=alt.Axis(title="Average Sentiment")),
            y=alt.Y('title_label:N', sort=alt.SortField('created_utc'), axis=alt.Axis(title="")),
            color='match_result:N',
            tooltip=['sentiment',
                     'most_positive_comment',
                     'most_negative_comment',
                     'comment_count']
            ).properties(
                width=700
            ))


@st.cache
def build_redditor_level_df(_df):
    df = _df.copy()
    df['author_str'] = df['author'].astype(str)
    author_count = df.groupby(["author_str"]
                              ).filter(lambda g: g['post_id'].nunique() > 5
                              ).groupby(["author_str", "match_result"]
                              ).nunique()

    author_result_count = author_count.reset_index().pivot(index="author_str", columns="match_result", values="post_id").fillna(0)
    author_result_count['total'] = author_result_count['lost'] + author_result_count['tie'] + author_result_count['won']
    author_result_count['won_prop'] = author_result_count['won']/author_result_count['total']
    author_result_count['lost_prop'] = author_result_count['lost']/author_result_count['total']
    author_result_count['tie_prop'] =  author_result_count['tie']/author_result_count['total']
    author_avg_sentiment = df[['sentiment', "score", 'author_str']].groupby("author_str").mean()[["sentiment", "score"]]

    author_df = author_result_count.merge(author_avg_sentiment, left_index=True, right_index=True)

    return author_df

redditor_df = build_redditor_level_df(comments_df)

st.write("""
         ### Finding 2: Negative commenters tend to post more often when we don't win

        Let's look at redditors that comment often in Post Match Threads (in at least 5 different threads). Among those,
        we see a reliationship that if you post more often in matches that we win, it is more likely that
        your average comment is positive.
         """)

reg_chart = alt.Chart(redditor_df.reset_index().rename(columns={'author_str': "redditor"})).mark_circle(size=50).encode(
    x=alt.X('won_prop', axis=alt.Axis(title="Percentage of posts they commented that were wins")),
    y=alt.Y('sentiment', axis=alt.Axis(title="Average comment sentiment")),
    tooltip=["redditor"]
)
reg_chart = reg_chart + reg_chart.transform_regression('won_prop', 'sentiment').mark_line()
st.altair_chart(reg_chart.properties(
                            width=600,
                            height=400,
                            title='Relationship between commenting in wins and average comment sentiment'


                        ).configure_title(

                        )
                )

st.write("""
Similarly, we see that the redditor with worst sentiment score tend to comment more often when we tie or lose.
**The table below shows the 5 redditors with the worst average sentiment**, all of these users comment in less than 37% of Post-Match threads
that were wins.

**This doesn't mean these users are toxic. For example, some of them might be using stronger
language which VADER might weight as more negative.**
         """)
worst_10_redditors = redditor_df[redditor_df["sentiment"] < 0].sort_values('sentiment').head(5)
worst_10_redditors.index = "/u/" + worst_10_redditors.index
worst_10_redditors['won_prop'] = (worst_10_redditors['won_prop']*100).astype(int).astype(str) + "%"
st.table(worst_10_redditors[['sentiment', 'won_prop']].rename(columns={'won_prop':
                                                                         "Percentage of posts they commented that were wins",
                                                                         'sentiment': "Average Sentiment"}))

st.write("""
         Also in the graph we can see that **there is a group of user that never comment when we win**:
         """)
never_win = redditor_df[redditor_df["won_prop"] == 0].sort_values('sentiment').head(5)
never_win.index = "/u/" + never_win.index
never_win['won_prop'] = (never_win['won_prop']*100).astype(int).astype(str) + "%"
st.table(never_win[['sentiment', 'won_prop']].rename(columns={'won_prop':
                                                                         "Percentage of posts they commented that were wins",
                                                                         'sentiment': "Average Sentiment"}))

st.write("""
         ## Explore the data

         ### Find out the sentiment of your comments

         This will show the top level comments you have made in Post Match Threads.

         Just remember that the score is not always a good measure of how positive or negative the
         comment is, but it works well on average.
        """)
selected_user = st.selectbox('Username', sorted(comments_df['author_str'].unique()))
user_comments = comments_df[comments_df['author_str'] == selected_user]
st.table(user_comments[["title_label", "body", "sentiment", "author_str"]].set_index("author_str"))

st.write("""
         ### Explore the 5 most negatives comments for each game
         """)

# Match select box
selected_match = st.selectbox('Match', sorted(matches_df['match_date_title']))

post_id = matches_df[matches_df['match_date_title'] == selected_match]["post_id"]
post_id = post_id.iloc[0]
filtered_comments = comments_df[comments_df.post_id == post_id
                               ].sort_values(by='sentiment'
                               )[['sentiment','body']].assign(hack='').set_index('hack')
maximum_comment_lenght = st.slider("Max comment length",
                                   value=300,
                                   min_value=min(filtered_comments.body.str.len()),
                                   max_value=max(filtered_comments.body.str.len()))
filtered_comments = filtered_comments[filtered_comments['body'].str.len() < maximum_comment_lenght]
st.table(filtered_comments.head())
