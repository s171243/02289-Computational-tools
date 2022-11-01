class User():
    def __init__(self, u):
        self.uuid = u['uuid']
        self.gender = u['gender']
        self.points = u['points']  # Enery points028
        self.badges_cnt = u['badges_cnt']
        self.first_login = u['first_login_date_TW']
        self.user_grade = u['user_grade']
        self.user_city = u['user_city']
        self.has_teacher = u['has_teacher_cnt']
        self.is_self_coach = u['is_self_coach']
        self.has_student_cnt = u['has_student_cnt']
        self.belongs_to_class_cnt = u['belongs_to_class_cnt']
        self.has_class_cnt = u['has_class_cnt']


class Problem():
    def __init__(self, pr):
        self.timestamp_TW = pr['timestamp_TW']
        self.uuid = pr['uuid']
        self.ucid = pr['ucid']
        self.upid = pr['upid']
        self.problem_number = pr['problem_number']
        self.exercise_problem_repeat_session = pr['exercise_problem_repeat_session']
        self.is_correct = pr['is_correct']
        self.total_sec_taken = pr['total_sec_taken'],
        self.total_attempt_cnt = pr['total_attempt_cnt']
        self.used_hint_cnt = pr['used_hint_cnt']
        self.is_hint_used = pr['is_hint_used']
        self.is_downgrade = pr['is_downgrade']
        self.is_upgrade = pr['is_upgrade']
        self.level = pr['level']


class Content():
    def __init__(self, ex):
        self.ucid = ex['ucid']
        self.content_pretty_name = ex['content_pretty_name']
        self.content_kind = ex['content_kind']
        self.difficulty = ex['difficulty']
        self.subject = ex['subject']
        self.learning_stage = ex['learning_stage']
        self.level1_id = ex['level1_id']
        self.level2_id = ex['level2_id']
        self.level3_id = ex['level3_id']
        self.level4_id = ex['level4_id']
