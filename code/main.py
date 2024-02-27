# 수면 Report_배포 코드 작성중

import plotly.graph_objects as go
import pandas as pd
import streamlit as st
import os
from datetime import datetime, timedelta
from langchain.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
#import mysql.connector
import pymysql
from langchain_core.output_parsers import StrOutputParser 
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationSummaryMemory
import tiktoken

# 경로
current_dir=os.path.dirname(__file__)
chatbot_dir=os.path.dirname(current_dir)
data_dir=os.path.join(chatbot_dir, 'Data', 'vectordbbasicparenting_faiss_index')
#db_path=os.path.join(data_dir, "index.faiss")

icon_path=os.path.join(chatbot_dir, 'Data', 'icon')
pic_path=os.path.join(icon_path, "Dr.COCO.png")
pic2_path = os.path.join(icon_path, "baby.png")

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['TOKENIZERS_PARALLELISM'] = st.secrets['TOKENIZERS_PARALLELISM']


# ChatOpenAI 모델 인스턴스 생성
chat_model = ChatOpenAI(temperature=0.1,
                #  max_tokens=800,  
                #  model_name='gpt-4-1106-preview',
                model_name="gpt-4-turbo-preview",
                #  streaming=True,              
                #  callbacks=[StreamingStdOutCallbackHandler()]
                )


# parser 추가
parser = StrOutputParser()

tokenizer=tiktoken.get_encoding("cl100k_base")

def titoken_len(text):
    tokens=tokenizer.encode(text)
    return len(tokens)

# 임베딩 초기화 (자연어 -> vector 변환)
ko=HuggingFaceEmbeddings(
    model_name='jhgan/ko-sbert-nli', #사용할 모델명
    model_kwargs={'device':'cpu'}, #모델에 전달할 키워드 인수
    encode_kwargs={'normalize_embeddings':True} #모델의 encode 메소드를 호출할 때 전달할 키워드 인수
)

memory = ConversationSummaryMemory(
    llm=chat_model,
    max_token_limit=80,
    memory_key="chat_history",
    human_prefix="### Friend",
    ai_prefix='### AI',
    output_key='answer',
    return_messages=True
)

category = ""




template="""
### You are an assistant who helps parents with parenting. 
# Answer questions using only the following context. If you don't know the answer, say you don't know and do not make it up: {context}
Answer questions in a helpful manner and engage in conversation while doing so. 
If asked about greetings, respond in a conversational way, but if you don't know the exact answer, say you don't know.
{chat_history}
### Friend : {question}
### AI: """.strip()

prompt=PromptTemplate(input_variables=['question','context','chat_history'], template=template)

#selected_db_path = os.path.join('COCOCHAT_SLEEPREPORT', 'ChatBot', 'Data', 'vectordbbasicparenting_faiss_index' )
selected_vector_db = FAISS.load_local(data_dir, ko)
retriever = selected_vector_db.as_retriever(search_type="similarity", search_kwargs={'k':3})
#print(selected_db_path)
   
chain=ConversationalRetrievalChain.from_llm(
    llm=chat_model,
    # chain_type="stuff",
    retriever=retriever,
    memory=memory,
    combine_docs_chain_kwargs={'prompt':prompt},
    return_source_documents=True,
)

# 분 단위의 정수를 'OO 시간 OO분' 형식으로 변환하는 함수
def format_minutes(minutes):
    hours = int(minutes // 60)
    minutes = round((minutes % 60) / 5) * 5  # 5분 단위로 올림
    if minutes == 60:  # 올림으로 인해 분이 60이 되면 시간을 1 증가시키고 분을 0으로 설정
        hours += 1
        minutes = 0
    return f"{hours}시간 {minutes}분"

# Mysql 연결
def connect_to_database():
    return pymysql.connect(
        host='127.0.0.1',
        port = '3306'
        user='guest',
        password='0000',
        database='ChatbotTest',
        cursorclass=pymysql.cursors.DictCursor
    )

# 주간 데이터 가져오기
def fetch_weekly_sleep_data():
    mydb = connect_to_database()
    with mydb.cursor() as cursor:
        week_ago = datetime.now() - timedelta(days=7)
        week_ago_str = week_ago.strftime('%Y-%m-%d')
        today_str = datetime.now().strftime('%Y-%m-%d')

        query = """
        SELECT date, time_daysleep, time_nightsleep, time_totalsleep, num_daysleep, child_age, child_name
        FROM BabySleepRecord
        WHERE date BETWEEN %s AND %s
        ORDER BY date ASC;
        """
        cursor.execute(query, (week_ago_str, today_str))
        result = cursor.fetchall()
    mydb.close()
    return pd.DataFrame(result)

# 색상 코드
day_sleep_color = '#fdc004'  
night_sleep_color = '#e86f0d' 
gray_color = '#595959'


# 시각화 함수
def visualize_sleep_data(df):
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%m.%d')
    fig = go.Figure()

    # 낮잠 시간 가로 막대 그래프 추가
    fig.add_trace(go.Bar(
        y=df['date'], 
        x=df['time_daysleep']/60, 
        name='낮잠 시간',
        marker_color=day_sleep_color,
        orientation='h',
        text = df['num_daysleep'].astype(str) + '회',
        textposition='inside',
        insidetextanchor='start'
    ))

    # 밤잠 시간 가로 막대 그래프 추가
    fig.add_trace(go.Bar(
        y=df['date'], 
        x=df['time_nightsleep']/60, 
        name='밤잠 시간',
        marker_color=night_sleep_color,
        orientation='h',
    ))
    # 총 수면 시간 텍스트 추가
    for i, row in df.iterrows():
        fig.add_annotation(
            x=row['time_totalsleep']/60,
            y=row['date'],
            text=f"{row['time_totalsleep']/60:.1f}h",
            showarrow=False,
            font=dict(size=12, color="black"),
            xanchor="left",
            yanchor="middle"
        )

    # 막대를 그룹화하도록 barmode를 'stack'으로 변경
    fig.update_layout(
        title='동동이의 주간 수면 시간',
        xaxis=dict(
            title='수면 시간 (시간)',
            showline=True,
            linewidth=2,
            linecolor='black'
        ),
        yaxis=dict(
            title='날짜',
            autorange='reversed',
            showline=True,
            linewidth=2,
            linecolor='black',
            tickmode='array',
            tickvals=df['date'],
            ticktext=df['date']
        ),
        barmode='stack'
    )

    return fig


# 평균 수면 시간 계산
def calculate_sleep_averages(df):
    avg_day_sleep = df['time_daysleep'].mean()
    avg_total_sleep = df['time_totalsleep'].mean()
    
    return avg_day_sleep, avg_total_sleep

# 또래 평균 수면 시간과 비교하는 막대 그래프 시각화 함수
def visualize_sleep_comparison(df, peer_avg_day_sleep=3, peer_avg_night_sleep=11, peer_avg_total_sleep=14):
    avg_day_sleep = df['time_daysleep'].mean() / 60
    avg_night_sleep = df['time_nightsleep'].mean()/60
    avg_total_sleep = df['time_totalsleep'].mean() / 60

    categories = ['낮잠 시간', '밤잠 시간', '총 수면 시간']
    your_sleep = [avg_day_sleep, avg_night_sleep, avg_total_sleep]
    peer_sleep = [peer_avg_day_sleep, peer_avg_night_sleep, peer_avg_total_sleep]

    fig = go.Figure(data=[
        go.Bar(name='동동이', x=categories, y=your_sleep, marker_color=day_sleep_color, width=0.4, text=[f"{x:.1f}h" for x in your_sleep], textposition='auto'),
        go.Bar(name='또래 평균', x=categories, y=peer_sleep, marker_color=gray_color, width=0.4, text=[f"{x:.1f}h" for x in peer_sleep], textposition='auto')
    ])
    fig.update_layout(
        title='수면 시간 비교',
        #xaxis=dict(title='카테고리'),
        yaxis=dict(title='시간 (시간)'),
        barmode='group'
    )
    return fig


# 평균 수면 시간 챗봇에 전달 + 질문
def ask_chatbot_about_sleep_averages(avg_day_sleep, avg_total_sleep, child_age, child_name):
    avg_day_sleep_formatted = format_minutes(avg_day_sleep)
    avg_total_sleep_formatted = format_minutes(avg_total_sleep)
    
    chatbot_question = (
        f"나이가 {child_age}개월인 {child_name}의 일주일 평균 낮잠 시간은 {avg_day_sleep_formatted}, "
        f"평균 수면 시간은 {avg_total_sleep_formatted}입니다. 이 수면 패턴은 연령대에 적절한가요?"
    )
    
    chatbot_response = chain(chatbot_question)  # 챗봇에게 질문을 전달하고 응답을 받기.
    
    return chatbot_response['answer']  # 챗봇의 응답 텍스트를 반환

def show_analysis_in_box(response):
    # Streamlit의 markdown 기능을 사용하여 박스 스타일을 적용합니다.
    st.markdown(f"""
    <div style="
        border: 2px solid #595959;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    ">
        <h3>Dr.COCO의 분석:</h3>
        <p>{response}</p>
    </div>
    """, unsafe_allow_html=True)

            


def main():
    st.image(pic2_path, width=100)
    st.title("수면 Report")
    
    df = fetch_weekly_sleep_data()

    # 낮잠 및 밤잠 시간 정보 표시
    avg_day_sleep, avg_total_sleep = calculate_sleep_averages(df)
    avg_day_sleep_str = format_minutes(avg_day_sleep)
    avg_total_sleep_str = format_minutes(avg_total_sleep)
    #avg_night_sleep_str = format_minutes(avg_total_sleep - avg_day_sleep)

    
    if not df.empty:

        # 시각화
        fig_bar = visualize_sleep_data(df)
        st.plotly_chart(fig_bar, use_container_width=True)

        #st.write(f"**지난 평균 낮잠 시간: {avg_day_sleep_str}")
        #st.write(f"**지난주 평균 총 수면 시간: {avg_total_sleep_str}")

         # 스타일링된 텍스트 표시
        st.markdown(f"<div style='background-color:#ffffff; padding:10px; border-radius:5px; margin-top:10px;'>"
                    f"<h5 style='color:#595959;'><b>지난주 평균 낮잠 시간:</b> {avg_day_sleep_str}</h5>"
                    f"<h5 style='color:#595959;'><b>지난주 평균 총 수면 시간:</b> {avg_total_sleep_str}</h5>"
                    f"</div>", unsafe_allow_html=True)

        # 동동이와 또래 평균 수면 시간 비교 막대 그래프 시각화
        fig_comparison = visualize_sleep_comparison(df)
        st.plotly_chart(fig_comparison, use_container_width=True)

        # 챗봇의 주간 평균 수면 시간에 대한 분석을 요청하고 표시
        child_age = df['child_age']
        child_name = df['child_name']
        chatbot_response = ask_chatbot_about_sleep_averages(
            avg_day_sleep, avg_total_sleep, child_age, child_name
        )
        st.image(pic_path, width=100)
        show_analysis_in_box(chatbot_response)
        #st.write(f"**Dr.COCO의 분석:** {chatbot_response}")


    
    else:
        st.write("지난주의 수면 데이터가 없습니다.")

if __name__ == "__main__":
    main()