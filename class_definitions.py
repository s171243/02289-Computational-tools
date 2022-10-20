
class User():
    def __init__(self,df_row):
        self.uuid = df_row['uuid']
        self.gender = df_row['gender']
        self.points = df_row['points'] #Enery points028
        self.badges_cnt = df_row['badges_cnt']
        self.first_login = df_row['first_login_date_TW']
        self.user_grade = df_row['user_grade']
        self.user_city = df_row['user_city']
        self.has_teacher = df_row['has_teacher_cnt']
        self.is_self_coach = df_row['is_self_coach']
        self.has_student_cnt = df_row['has_student_cnt']
        self.belongs_to_class_cnt = df_row['belongs_to_class_cnt']
        self.has_class_cnt = df_row['has_class_cnt']

class Problem():
    def __init__(self, df_prob):
        self.timestamp_TW = df_prob['timestamp_TW']
        self.uuid = df_prob['uuid']
        self.ucid = df_prob['ucid']
        self.upid = df_prob['upid']
        self.problem_number = df_prob['problem_number']
        self.exercise_problem_repeat_session = df_prob['exercise_problem_repeat_session']
        self.is_correct = df_prob['is_correct']
        self.total_sec_taken = df_prob['total_sec_taken'],
        self.total_attempt_cnt = df_prob['total_attempt_cnt']
        self.used_hint_cnt = df_prob['used_hint_cnt']
        self.is_hint_used = df_prob['is_hint_used']
        self.is_downgrade = df_prob['is_downgrade']
        self.is_upgrade = df_prob['is_upgrade']
        self.level = df_prob['level']
class Content():
    def __init__(self,df_content):
        self.ucid = df_content['ucid']
        self.content_pretty_name = df_content['content_pretty_name']
        self.content_kind = df_content['content_kind']
        self.difficulty = df_content['difficulty']
        self.subject = df_content['subject']
        self.learning_stage = df_content['learning_stage']
        self.level1_id = df_content['level1_id']
        self.level2_id = df_content['level2_id']
        self.level3_id = df_content['level3_id']
        self.level4_id = df_content['level4_id']