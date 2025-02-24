# This files contains your custom actions which can be used to run
# custom Python code.
#
# See this guide on how to implement these action:
# https://rasa.com/docs/rasa/custom-actions


# This is a simple example for a custom action which utters "Hello World!"

from typing import Any, Text, Dict, List
import numpy as np
import re
#
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa.nlu.utils import write_json_to_file
from rasa_sdk.events import SlotSet

from natasha import (
    Segmenter,
    MorphVocab,
    NewsEmbedding,
    NewsMorphTagger,
    Doc
)

with open('/Users/ma.lunev/Documents/rasa_project/actions/stopWords.txt') as file:
    stops = [line.rstrip() for line in file]
#
#
# class ActionHelloWorld(Action):
#
#     def name(self) -> Text:
#         return "action_hello_world"
#
#     def run(self, dispatcher: CollectingDispatcher,
#             tracker: Tracker,
#             domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
#
#         dispatcher.utter_message(text="Hello World!")
#
#         return []

segmenter = Segmenter()
morph_vocab = MorphVocab()
embedding = NewsEmbedding()
morph_tagger = NewsMorphTagger(embedding)

# Функция для получения эмбеддингов слов
def get_word_embeddings_mean(text):
    # Создание документа
    doc = Doc(text.lower().replace('-', ' ').replace('?', ' ').replace('.', ' ').replace(',', ' ').replace(';', ' ').replace(':', ' '))
    
    # Токенизация
    doc.segment(segmenter)
    
    # Лемматизация и получение эмбеддингов
    doc.tag_morph(morph_tagger)
    
    embeddings = []
    mean = 0
    cnt = 0
    for token in doc.tokens:
        if not(token.text in stops):
            try:
                vector = embedding[token.text]
                embeddings.append((token.text, vector))
                if cnt==0:
                    mean = vector
                else:
                    mean += vector
                cnt+=1
            except KeyError:
                pass
    
    return mean/cnt

data_analyst_texts = ['''Я начал свою карьеру в качестве стажера в компании Яндекс, 
где работал над проектами, связанными с анализом пользовательских данных. После успешного завершения 
стажировки меня приняли на должность data analyst. В течение двух лет я занимался анализом поведения 
пользователей и разработкой отчетов для команды маркетинга. Я освоил SQL и Tableau, и теперь стремлюсь
 развивать свои навыки в области машинного обучения.''', '''После получения диплома в области прикладной 
 математики я устроился на работу в Mail.ru Group в качестве data analyst. За два года я анализировал 
 эффективность рекламных кампаний и создавал визуализации для команды. Я научился использовать Python 
 для обработки данных и пайтон стал экспертом в Excel. Моя цель — перейти на более высокую позицию, где смогу 
 управлять проектами и командой.''', '''Я начал свою карьеру в компании Сбер, где проработал два года в 
 роли data analyst. Моя основная задача заключалась в анализе финансовых показателей и подготовке отчетов 
 для клиентов. Я освоил инструменты BI, такие как Power BI, и научился работать с большими объемами 
 данных. Теперь я хочу углубить свои знания в области предиктивной аналитики и применять их в своей 
 работе.''', '''В течение двух лет я работал data analyst в стартапе Ozon, где занимался анализом 
 пользовательских данных и поведением клиентов. Я активно использовал SQL и R для анализа данных и 
 создания визуализаций. Мой опыт в стартапе пайтон научил меня быстро адаптироваться к изменениям и работать в 
 условиях неопределенности. Я стремлюсь развивать свои навыки в области аналитики больших данных.''', '''Я 
 начал свою карьеру в крупной IT-компании Тинькофф, где проработал два года в качестве data analyst. Моя 
 работа заключалась в пайтон анализе данных о продажах и клиентских предпочтениях, что помогло мне понять, как 
 данные могут влиять на бизнес-решения. Я освоил методы A/B-тестирования и научился работать с различными 
 аналитическими инструментами. В будущем я планирую углубить свои знания в области аналитики и разработки 
 алгоритмов.''','''Я начал свою карьеру в компании ВКонтакте в качестве стажера, а затем был принят на 
 должность data analyst. В течение двух лет я анализировал пользовательские данные и 
 готовил отчеты для команды маркетинга.''', '''В компании Lamoda я проработал два 
 года в качестве data analyst, анализируя пользовательские данные и поведение клиентов в 
 сфере электронной коммерции.''', '''После получения диплома я устроился в компанию Ростелеком 
 на позицию data analyst. За два года я занимался анализом эффективности услуг и создавал визуализации для команды.''']
data_analyst_embeddings = [get_word_embeddings_mean(x) for x in data_analyst_texts]

mlops_engineer_texts = ["Я начал свою карьеру в компании Яндекс в качестве стажера, а затем был принят на должность MLOps engineer. В течение двух лет я занимался развертыванием и поддержкой моделей машинного обучения для различных проектов.",
"После получения диплома я устроился в компанию Сбер на позицию MLOps engineer. За два года я работал над автоматизацией процессов развертывания моделей и оптимизацией их производительности.",
"Я начал работать в компании Mail.ru Group в роли MLOps engineer, где занимался интеграцией моделей в продакшн и обеспечением их стабильной работы на протяжении двух лет.",
"В компании Ozon я проработал два года в качестве MLOps engineer, разрабатывая и поддерживая инфраструктуру для моделей машинного обучения в сфере электронной коммерции.",
"Я начал свою карьеру в компании Тинькофф, где проработал два года в роли MLOps engineer, занимаясь мониторингом и улучшением моделей, используемых в финансовых продуктах.",
"Я начал свою карьеру в компании Ростелеком в качестве MLOps engineer, где в течение двух лет занимался разработкой и внедрением CI/CD процессов для моделей машинного обучения, что значительно ускорило их развертывание.",
"После окончания университета я устроился в компанию ВКонтакте на позицию MLOps engineer. За два года я работал над оптимизацией инфраструктуры для обработки больших данных и интеграцией моделей в существующие системы.",
"Я проработал два года в компании Альфа-Банк в роли MLOps engineer, где занимался автоматизацией процессов мониторинга и тестирования моделей, что позволило повысить их надежность и производительность.",
"В компании Lamoda я начал свою карьеру как MLOps engineer, где в течение двух лет разрабатывал и поддерживал системы для обработки и анализа данных о покупках, что способствовало улучшению клиентского опыта.",
"Я начал работать в компании Тинькофф Инвестиции в роли MLOps engineer, где за два года занимался внедрением решений для масштабирования моделей машинного обучения и обеспечением их бесперебойной работы в продакшене."]
mlops_engineer_embeddings = [get_word_embeddings_mean(x) for x in mlops_engineer_texts]

project_manager_texts = ["Я начал свою карьеру в компании Яндекс в качестве помощника проектного менеджера, а затем был повышен до проектного менеджера. За четыре года я успешно управлял несколькими проектами по разработке новых функций для поисковой системы.",
"После окончания университета я устроился в компанию Сбер на позицию project manager. В течение четырех лет я координировал команды разработчиков и дизайнеров, обеспечивая успешное выполнение проектов в срок и в рамках бюджета.",
"Я проработал четыре года в компании Mail.ru Group в роли проектного менеджера, где занимался управлением проектами по созданию новых продуктов и улучшению существующих, что способствовало росту пользовательской базы.",
"В компании Ozon я начал свою карьеру как project manager, где в течение четырех лет управлял проектами по внедрению новых технологий в сфере электронной коммерции, что значительно повысило эффективность бизнес-процессов.",
"Я работал в компании Тинькофф в роли проектного менеджера, где за четыре года успешно реализовал несколько проектов по разработке финансовых приложений, что позволило улучшить клиентский опыт и увеличить доходы компании.",
"В компании ВКонтакте я начал свою карьеру как project manager, где в течение четырех лет управлял проектами по разработке новых функций социальной сети, что способствовало увеличению вовлеченности пользователей.",
"Я проработал четыре года в Альфа-Банке в роли проектного менеджера, где занимался координацией проектов по внедрению новых банковских услуг и оптимизации внутренних процессов, что позволило повысить уровень обслуживания клиентов.",
"В компании Ростелеком я начал свою карьеру как project manager, где в течение четырех лет управлял проектами по цифровизации услуг, что значительно улучшило качество обслуживания клиентов.",
"Я работал в компании Lamoda в роли проектного менеджера, где за четыре года успешно реализовал проекты по улучшению логистики и оптимизации процессов обработки заказов, что способствовало росту продаж.",
"В компании Тинькофф Инвестиции я начал свою карьеру как project manager, где в течение четырех лет управлял проектами по разработке инвестиционных платформ, что позволило привлечь новых клиентов и увеличить объемы торгов."]
project_manager_embeddings = [get_word_embeddings_mean(x) for x in project_manager_texts]

data_engineer_texts = ["Я начал свою карьеру в компании Яндекс в качестве стажера по обработке данных, а затем был повышен до инженера данных. За два года я работал над проектами по оптимизации обработки больших объемов данных, разрабатывая и внедряя ETL-процессы, что улучшило алгоритмы поиска и повысило скорость обработки запросов.",
"После окончания университета я устроился в компанию Сбер на позицию аналитика данных. В течение двух лет я занимался разработкой ETL-процессов и интеграцией данных из различных источников, а также создавал дашборды для визуализации данных, что способствовало улучшению аналитики и принятию обоснованных бизнес-решений.",
"Я проработал два года в компании Mail.ru Group в роли инженера по данным, где занимался построением и поддержкой хранилищ данных. Я также оптимизировал запросы к базе данных и разрабатывал скрипты для автоматизации обработки данных, что позволило команде более эффективно анализировать пользовательские метрики.",
"В компании Ozon я начал свою карьеру как специалист по обработке данных, где в течение двух лет разрабатывал решения для автоматизации сбора и анализа данных. Я внедрил системы мониторинга качества данных, что значительно повысило скорость принятия решений и улучшило точность отчетности.",
"Я работал в компании Тинькофф в роли data engineer, где за два года успешно реализовал проекты по интеграции данных из различных систем. Я также занимался разработкой и оптимизацией хранилищ данных, что позволило улучшить качество отчетности и аналитики, а также ускорить доступ к данным и качество данных для бизнес-пользователей.",
"В компании ВКонтакте я начал свою карьеру как инженер по данным, где в течение двух лет занимался разработкой и оптимизацией процессов обработки данных. Я работал над проектами по анализу пользовательского поведения и внедрению машинного обучения для персонализации контента, что способствовало улучшению пользовательского опыта.",
"Я проработал два года в Альфа-Банке в роли специалиста по данным, где занимался анализом и обработкой больших данных для разработки новых финансовых продуктов. Я также разрабатывал модели для прогнозирования рисков и оптимизации кредитных решений, что позволило повысить конкурентоспособность банка.",
"В компании Ростелеком я начал свою карьеру как data engineer, где в течение двух лет разрабатывал и поддерживал системы для обработки и хранения данных. Я внедрил процессы для обеспечения качества данных и автоматизации отчетности, что значительно улучшило качество обслуживания клиентов.",
"Я работал в компании Lamoda в роли инженера по данным, где за два года успешно реализовал проекты по оптимизации логистики с использованием аналитики данных. Я разрабатывал алгоритмы для прогнозирования спроса и оптимизации запасов, что способствовало росту продаж и снижению издержек.",
"В компании Тинькофф Инвестиции я начал свою карьеру как аналитик данных, где в течение двух лет занимался разработкой моделей для прогнозирования рыночных трендов. Я также работал над проектами по анализу больших данных для выявления инвестиционных возможностей, что позволило привлечь новых клиентов и увеличить объемы торгов."]
data_engineer_embeddings = [get_word_embeddings_mean(x) for x in data_engineer_texts]

data_scientist_texts = ["Я начал свою карьеру в компании Яндекс в качестве стажера по анализу данных, а затем был повышен до data scientist. За три года я работал над проектами по разработке моделей машинного обучения для прогнозирования пользовательского поведения, что позволило значительно улучшить персонализацию контента.",
"После окончания университета я устроился в компанию Сбер на позицию аналитика данных. В течение трех лет я занимался разработкой и внедрением алгоритмов для анализа больших данных, а также создавал дашборды для визуализации результатов, что способствовало принятию более обоснованных бизнес-решений.",
"Я проработал три года в компании Mail.ru Group в роли специалиста по данным, где занимался построением и оптимизацией моделей для анализа пользовательских метрик. Я также внедрил системы A/B-тестирования, что позволило команде улучшить качество продуктов на основе реальных данных.",
"В компании Ozon я начал свою карьеру как исследователь данных, где в течение трех лет разрабатывал решения для анализа покупательского поведения. Я использовал методы машинного обучения для сегментации клиентов, что значительно повысило эффективность маркетинговых кампаний.",
"Я работал в компании Тинькофф в роли data scientist, где за три года успешно реализовал проекты по прогнозированию финансовых рисков. Я разрабатывал модели для оценки кредитоспособности клиентов, что позволило улучшить качество кредитных решений и снизить уровень дефолтов.",
"В компании ВКонтакте я начал свою карьеру как аналитик данных, где в течение трех лет занимался разработкой и оптимизацией алгоритмов рекомендаций. Я работал над проектами по анализу пользовательского контента и внедрению машинного обучения для улучшения пользовательского опыта.",
"Я проработал три года в Альфа-Банке в роли data analyst, где занимался анализом больших данных для разработки новых финансовых продуктов. Я также разрабатывал модели для прогнозирования рыночных трендов, что позволило повысить конкурентоспособность банка.",
"В компании Ростелеком я начал свою карьеру как data scientist, где в течение трех лет разрабатывал и поддерживал системы для анализа данных. Я внедрил методы обработки естественного языка для анализа отзывов клиентов, что значительно улучшило качество обслуживания.",
"Я работал в компании Lamoda в роли специалиста по данным, где за три года успешно реализовал проекты по оптимизации логистики с использованием аналитики данных. Я разрабатывал модели для прогнозирования спроса и оптимизации запасов, что способствовало росту продаж и снижению издержек.",
"В компании Тинькофф Инвестиции я начал свою карьеру как исследователь данных, где в течение трех лет занимался разработкой моделей для анализа инвестиционных возможностей. Я также работал над проектами по анализу больших данных для выявления трендов на финансовых рынках, что позволило привлечь новых клиентов и увеличить объемы торгов."]
data_scientist_embeddings = [get_word_embeddings_mean(x) for x in data_scientist_texts]

project_manager_skillset = "Управление проектами, Коммуникационные навыки, Лидерство, Управление рисками, Планирование и организация, Аналитические навыки, Управление временем, Знание методологий управления проектами, Работа в команде, Финансовое управление, Управление изменениями, Использование инструментов управления проектами (например, Jira, Trello, Asana), Знание основ Agile и Scrum, Оценка и управление ресурсами, Создание и управление документацией проекта.Project Management, Communication Skills, Leadership, Risk Management, Planning and Organization, Analytical Skills, Time Management, Knowledge of Project Management Methodologies, Teamwork, Financial Management, Change Management, Use of Project Management Tools (e.g., Jira, Trello, Asana), Knowledge of Agile and Scrum Principles, Resource Estimation and Management, Project Documentation Creation and Management."
project_manager_skillset_embeddings = get_word_embeddings_mean(project_manager_skillset)

mlops_skillset = '''Управление жизненным циклом моделей (Model Lifecycle Management),  
Знание машинного обучения (Knowledge of Machine Learning),  
Разработка и развертывание моделей (Model Development and Deployment),  
Автоматизация процессов (Process Automation),  
Управление данными (Data Management),  
Инструменты для CI/CD (CI/CD Tools),  
Контейнеризация (Containerization, например, Docker),  
Оркестрация (Orchestration, например, Kubernetes),  
Мониторинг и логирование (Monitoring and Logging),  
Знание облачных платформ (Knowledge of Cloud Platforms, например, AWS, GCP, Azure),  
Скриптование (Scripting, например, Python, Bash),  
Работа с большими данными (Big Data Technologies, например, Hadoop, Spark),  
Обеспечение качества моделей (Model Quality Assurance),  
Соблюдение норм и стандартов (Compliance and Standards).
Model Lifecycle Management,  
Knowledge of Machine Learning,  
Model Development and Deployment,  
Process Automation,  
Data Management,  
CI/CD Tools,  
Containerization (e.g., Docker),  
Orchestration (e.g., Kubernetes),  
Monitoring and Logging,  
Knowledge of Cloud Platforms (e.g., AWS, GCP, Azure),  
Scripting (e.g., Python, Bash),  
Big Data Technologies (e.g., Hadoop, Spark),  
Model Quality Assurance'''
mlops_skillset_embeddings = get_word_embeddings_mean(mlops_skillset)

data_engineer_skillset='''Проектирование и разработка архитектуры данных,  
Управление данными (Data Management),  
Интеграция данных (Data Integration),  
Обработка и трансформация данных (Data Processing and Transformation),  
Знание SQL и NoSQL баз данных,  
Работа с большими данными (Big Data Technologies, например, Hadoop, Spark),  
Создание и управление ETL-процессами (ETL Processes),  
Оптимизация производительности баз данных (Database Performance Tuning),  
Знание облачных платформ (Knowledge of Cloud Platforms, например, AWS, GCP, Azure),  
Скриптование (Scripting, например, Python, Java, Scala),  
Мониторинг и управление качеством данных (Data Quality Monitoring and Management),  
Обеспечение безопасности данных (Data Security),  
Знание инструментов для работы с потоками данных (Data Streaming Tools, например, Apache Kafka),  
Соблюдение норм и стандартов (Compliance and Standards).
Data Architecture Design and Development,  
Data Management,  
Data Integration,  
Data Processing and Transformation,  
Knowledge of SQL and NoSQL Databases,  
Big Data Technologies (e.g., Hadoop, Spark),  
ETL Processes Creation and Management,  
Database Performance Tuning,  
Knowledge of Cloud Platforms (e.g., AWS, GCP, Azure),  
Scripting (e.g., Python, Java, Scala),  
Data Quality Monitoring and Management,  
Data Security,  
Data Streaming Tools (e.g., Apache Kafka)'''
data_engineer_skillset_embeddings = get_word_embeddings_mean(data_engineer_skillset)

analyst_skillset='''Анализ данных (Data Analysis), Знание SQL (Knowledge of SQL), Визуализация данных (Data Visualization), Статистический анализ (Statistical Analysis), Работа с инструментами BI (Business Intelligence Tools, например, Tableau, Power BI), Обработка и очистка данных (Data Cleaning and Preparation), Знание Excel (Knowledge of Excel), Интерпретация результатов (Results Interpretation), Знание языков программирования для анализа данных (Knowledge of Programming Languages for Data Analysis, например, Python, R), Коммуникационные навыки (Communication Skills), Понимание бизнес-процессов (Understanding of Business Processes), Работа с большими данными (Big Data Technologies, например, Hadoop, Spark), 
Data Analysis, Knowledge of SQL, Data Visualization, Statistical Analysis, Business Intelligence Tools (e.g., Tableau, Power BI), Data Cleaning and Preparation, Knowledge of Excel, Results Interpretation, Knowledge of Programming Languages for Data Analysis (e.g., Python, R), Communication Skills, Understanding of Business Processes, Big Data Technologies (e.g., Hadoop, Spark)'''
analyst_skillset_embeddings = get_word_embeddings_mean(analyst_skillset)

scientist_skillset = '''Анализ данных (Data Analysis), Знание языков программирования (Knowledge of Programming Languages, например, Python, R), Машинное обучение (Machine Learning), Статистический анализ (Statistical Analysis), Визуализация данных (Data Visualization), Обработка и очистка данных (Data Cleaning and Preparation), Работа с большими данными (Big Data Technologies, например, Hadoop, Spark), Знание SQL (Knowledge of SQL), Понимание алгоритмов и структур данных (Understanding of Algorithms and Data Structures), Коммуникационные навыки (Communication Skills), Понимание бизнес-проблем (Understanding of Business Problems), Разработка моделей (Model Development), Тестирование и валидация моделей (Model Testing and Validation).
Data Analysis, Knowledge of Programming Languages (e.g., Python, R), Machine Learning, Statistical Analysis, Data Visualization, Data Cleaning and Preparation, Big Data Technologies (e.g., Hadoop, Spark), Knowledge of SQL, Understanding of Algorithms and Data Structures, Communication Skills, Understanding of Business Problems, Model Development, Model Testing and Validation.'''
scientist_skillset_embeddings = get_word_embeddings_mean(scientist_skillset)

def decode(wanted_position) -> Text:
    slot_value = wanted_position
    if re.search('проект|проджект|project|перв|1', slot_value.lower()):
        return 'project_manager'
    elif re.search('data eng|дата инженер|инженер данн|втор|2', slot_value.lower()):
        return 'data_engineer'
    elif re.search('аналитик|дата аналит|аналитик данн|data analyst|трет|3', slot_value.lower()):
        return 'data_analyst'
    elif re.search('саентист|scientist|чет|4', slot_value.lower()):
        return 'data_scientist'
    elif re.search('млопс|девопс|mlops|пят|пять|5', slot_value.lower()):
        return 'mlops_engineer'
    else:
        return 'idk'

class ContactsAction(Action):
    def name(self) -> Text:
        return "action_save_contacts"
  
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        write_json_to_file('contacts.json', tracker.slots)
        print(decode(tracker.get_slot('wanted_position')))
        return {}

class checkCandidate(Action):
    def name(self) -> Text:
        return "action_check_candidate"
  
    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> Dict[Text, Any]:
        candidate_ok = True
        pos = decode(tracker.get_slot('wanted_position'))
        old_jobs = tracker.get_slot('positions')
        skills = tracker.get_slot('work_skills')
        xp = tracker.get_slot('work_experience')

        print('got slots')
        
        if (pos=='project_manager'):
            old_jobs_emb = project_manager_embeddings
            skills_emb = project_manager_skillset_embeddings
        elif (pos=='data_engineer'):
            old_jobs_emb = data_engineer_embeddings
            skills_emb = data_engineer_skillset_embeddings
        elif (pos=='data_analyst'):
            old_jobs_emb = data_analyst_embeddings
            skills_emb = analyst_skillset_embeddings
        elif (pos=='data_scientist'):
            old_jobs_emb = data_scientist_embeddings
            skills_emb = scientist_skillset_embeddings
        elif (pos=='mlops_engineer'):
            old_jobs_emb = mlops_engineer_embeddings
            skills_emb = mlops_skillset_embeddings
        else:
            cadidate_ok = False
        
        print(pos)
        if candidate_ok:
            v = get_word_embeddings_mean(old_jobs)
            scores = np.array([np.dot(v, x)/(np.linalg.norm(v)*np.linalg.norm(x)) for x in old_jobs_emb])
            if scores.max()<0.5:
                candidate_ok=False
                print("не прошел, расстояние от идеала по должностям: " + str(scores))
            
            v = get_word_embeddings_mean(skills)
            x = skills_emb
            scores = np.dot(v, x)/(np.linalg.norm(v)*np.linalg.norm(x))
            if scores.max()<0.15:
                candidate_ok=False
                print("не прошел, расстояние от идеала по скиллам: " + str(scores))
            
            if int(xp)<2:
                candidate_ok=False
                print(xp)
        
        SlotSet('candidate_ok', candidate_ok)
        print(candidate_ok)

        if candidate_ok:
            dispatcher.utter_message("Поздравляем! Вы прошли первый этап собеседований, с вами обязательно свяжется наш рекрутер.")
        else:
            dispatcher.utter_message("Спасибо за ответы, обязательно рассмотрим вашу заявку!")
        return {}