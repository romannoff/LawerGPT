import streamlit as st
import uuid
from datetime import datetime

# Настройка страницы
st.set_page_config(
    page_title="Юридическая RAG система",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Инициализация session state
if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ""
if 'chats' not in st.session_state:
    st.session_state.chats = {}
if 'current_chat_id' not in st.session_state:
    st.session_state.current_chat_id = None
if 'settings' not in st.session_state:
    st.session_state.settings = {
        'mode': 'standard',
        'router_enabled': True,
        'router_temperature': 0.7,
        'db_enabled': True,
        'chunks_count': 5,
        'reranker_enabled': True,
        'reranker_temperature': 0.5,
        'chain_of_thoughts': False,
        'multi_agent': False
    }

# Стили CSS
st.markdown("""
<style>
    .chat-row {
        display: flex;
        width: 100%;
        margin-bottom: 0.5rem;
    }
    .chat-row.user {
        justify-content: flex-end;
    }
    .chat-row.agent {
        justify-content: flex-start;
    }
    .chat-bubble {
        max-width: 70%;
        padding: 0.75rem 1rem;
        border-radius: 1rem;
        position: relative;
        line-height: 1.4;
    }
    .chat-bubble.user {
        background-color: #acf;
        color: #000;
        border-bottom-right-radius: 0;
    }
    .chat-bubble.agent {
        background-color: #eee;
        color: #000;
        border-bottom-left-radius: 0;
    }
</style>
""", unsafe_allow_html=True)

# Функции для работы с чатами
def create_new_chat():
    chat_id = str(uuid.uuid4())
    st.session_state.chats[chat_id] = {
        'id': chat_id,
        'name': f"Чат {len(st.session_state.chats) + 1}",
        'created_at': datetime.now(),
        'messages': []
    }
    st.session_state.current_chat_id = chat_id

def delete_chat(chat_id):
    if chat_id in st.session_state.chats:
        del st.session_state.chats[chat_id]
        if st.session_state.current_chat_id == chat_id:
            st.session_state.current_chat_id = None

def clear_chat_history():
    cid = st.session_state.current_chat_id
    if cid:
        st.session_state.chats[cid]['messages'] = []

# Заглушка RAG
def get_rag_response(query, settings):
    return {
        'answer': f"Ответ на ваш юридический вопрос: '{query}' ...",
        'contexts': [
            {'title': 'ГК РФ, ст.123', 'content': 'Пример...', 'relevance_score': 0.95},
            {'title': 'Постановление Пленума ВС РФ, п.15', 'content': 'Пример...', 'relevance_score': 0.87},
            {'title': 'ФЗ-123, ст.45', 'content': 'Пример...', 'relevance_score': 0.82},
        ]
    }

# Функция обработки ввода
def process_input():
    q = st.session_state['user_input'].strip()
    if q:
        current_chat = st.session_state.chats.get(st.session_state.current_chat_id)
        if current_chat:
            # Добавляем сообщение пользователя
            current_chat['messages'].append({
                'role': 'user',
                'content': q,
                'timestamp': datetime.now()
            })
            # Получаем ответ
            with st.spinner("Ищем ответ..."):
                resp = get_rag_response(q, st.session_state.settings)
            # Добавляем ответ ассистента
            current_chat['messages'].append({
                'role': 'assistant',
                'content': resp['answer'],
                'contexts': resp['contexts'],
                'timestamp': datetime.now()
            })
            # Очищаем поле ввода
            st.session_state['user_input'] = ""

# Левая боковая панель
with st.sidebar:
    if st.button("➕ Новый чат", use_container_width=True):
        create_new_chat()
        st.rerun()

    st.markdown("---")
    for chat_id, chat in st.session_state.chats.items():
        cols = st.columns([8,1])
        with cols[0]:
            if st.button(chat['name'],
                         key=f"chat_{chat_id}",
                         use_container_width=True,
                         type="primary" if chat_id == st.session_state.current_chat_id else "secondary"):
                st.session_state.current_chat_id = chat_id
                st.rerun()
        with cols[1]:
            if st.button("🗑️",
                         key=f"del_{chat_id}",
                         use_container_width=True):
                delete_chat(chat_id)
                st.rerun()

# Основная область и панель настроек
col_main, col_settings = st.columns([3, 1])

with col_main:
    st.markdown("## ⚖️ Юридическая RAG система")

    # Выбор или создание чата
    if not st.session_state.current_chat_id:
        if not st.session_state.chats:
            create_new_chat()
        else:
            st.session_state.current_chat_id = next(iter(st.session_state.chats))
    current_chat = st.session_state.chats.get(st.session_state.current_chat_id)

    if current_chat:
        # Кнопка очистки истории
        if st.button("🧹 Очистить диалог", use_container_width=True):
            clear_chat_history()

        # Отображение истории чата
        for msg in current_chat['messages']:
            role = msg['role']
            content = msg['content']
            row_cls = "chat-row user" if role == "user" else "chat-row agent"
            bub_cls = "chat-bubble user" if role == "user" else "chat-bubble agent"
            st.markdown(f"""
                <div class="{row_cls}">
                <div class="{bub_cls}">
                    {content}
                </div>
                </div>
            """, unsafe_allow_html=True)

        # Поле ввода и кнопка отправки
        st.text_area("Ваш вопрос:", height=100, key="user_input")
        if st.button("Отправить", on_click=process_input, use_container_width=True):
            pass

with col_settings:
    st.markdown("### ⚙️ Настройки")

    with st.expander("Основные параметры", expanded=True):
        mode = st.radio(
            "Режим работы:",
            options=['fast', 'standard', 'deep'],
            format_func=lambda x: {'fast': '⚡ Быстрый', 'standard': '⚖️ Стандартный', 'deep': '🧠 Глубокий'}[x],
            index=['fast', 'standard', 'deep'].index(st.session_state.settings['mode']),
            horizontal=True,
            key="mode_selector"  # Добавляем уникальный ключ для радиокнопок
        )
        if mode != st.session_state.settings['mode']:
            st.session_state.settings['mode'] = mode
            st.rerun()

    with st.expander("Настройки разработчика", expanded=False):
        st.checkbox("Включить роутер",
                    value=st.session_state.settings['router_enabled'],
                    key="router_enabled")
        if st.session_state.settings['router_enabled']:
            st.slider("Температура роутера",
                      0.0,1.0,st.session_state.settings['router_temperature'],step=0.1,
                      key="router_temperature")

        st.markdown("---")
        st.checkbox("Включить БД",
                    value=st.session_state.settings['db_enabled'],
                    key="db_enabled")
        if st.session_state.settings['db_enabled']:
            st.number_input("Число чанков",1,20,
                            st.session_state.settings['chunks_count'],
                            key="chunks_count")

        st.markdown("---")
        st.checkbox("Включить реранкер",
                    value=st.session_state.settings['reranker_enabled'],
                    key="reranker_enabled")
        if st.session_state.settings['reranker_enabled']:
            st.slider("Температура реранкера",
                      0.0,1.0,st.session_state.settings['reranker_temperature'],step=0.1,
                      key="reranker_temperature")

        st.markdown("---")
        st.checkbox("Chain of Thoughts", key="chain_of_thoughts")
        st.checkbox("Мультиагентный режим", key="multi_agent")

    if st.button("💾 Сохранить настройки", use_container_width=True):
        st.success("Настройки сохранены!")

# Футер
st.markdown("---")
st.markdown(
    '<p style="text-align:center;color:#888;">Юридическая RAG система v1.0</p>',
    unsafe_allow_html=True
)