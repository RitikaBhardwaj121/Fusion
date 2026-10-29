"""Database-free tests for the section-wise timetable pre-check.

Run from the FusionIIIT directory::

    python -m unittest applications.academic_procedures.test_section_precheck

The real ``compute_section`` and ``pick_offering`` are loaded from
academic_information/models.py as source (``ast``), so these tests exercise
the college's own section rules without Django settings or a database.
"""

import ast
import json
import os
import unittest
from types import SimpleNamespace

from applications.academic_procedures.solver import section_precheck as sp

APPS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _parse(relative_path):
    with open(os.path.join(APPS_DIR, relative_path), encoding='utf-8') as source:
        return ast.parse(source.read())


def _load_functions(relative_path, *names):
    tree = _parse(relative_path)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names), f'{names} not found in {relative_path}'
    namespace = {}
    exec(compile(ast.Module(nodes, []), relative_path, 'exec'), namespace)
    return [namespace[name] for name in names]


compute_section, pick_offering = _load_functions(
    'academic_information/models.py', 'compute_section', 'pick_offering')

CSE = 'Computer Science and Engineering'
ECE = 'Electronics and Communication Engineering'
TERM = sp.Term(2025, sp.EVEN)          # 2025-26 Even: offering year 2026


# ── Fixture builders ─────────────────────────────────────────────────────────

def batch(batch_id, name='B.Tech', discipline=CSE, acronym='CSE', year=2025,
          curriculum=None, semesters=8):
    return sp.BatchRecord(batch_id, name, discipline, acronym, year,
                          curriculum if curriculum is not None else batch_id, semesters)


def students(batch_id, count, programme='B.Tech', start=1, prefix='25BCS',
             section=None, semester=None):
    return [sp.StudentRecord(f'{prefix}{n:03d}', programme, batch_id, section, semester)
            for n in range(start, start + count)]


def course(course_id, code=None, lec=3, tut=1, prac=0, credit=None):
    if credit is None and None not in (lec, tut, prac):
        credit = lec + tut + prac // 2
    return sp.CourseRecord(course_id, code or f'C{course_id}', f'Course {course_id}',
                           credit, lec, tut, prac)


def slot(slot_id, curriculum, course_ids, semester=2, slot_type='Professional Core'):
    return sp.CourseSlotRecord(slot_id, curriculum, semester, f'Slot {slot_id}',
                               slot_type, tuple(course_ids))


def offering(offering_id, course_id, instructor, label=None, year=2026,
             semester_type=sp.EVEN):
    return sp.OfferingRecord(offering_id, course_id, instructor, year, semester_type, label)


def hall(room_id, capacity, room_type='Lecture Hall', active=True):
    return sp.RoomRecord(room_id, capacity, room_type, active)


ROOMS = [hall('LH1', 200), hall('LAB1', 200, 'Computer Lab')]


def run(batches, studs, slots, courses, offers, rooms=ROOMS, term=TERM,
        semesters=None, **kwargs):
    if semesters is None:      # by default every batch's semester exists
        semesters = {(b.curriculum_id, sem) for b in batches for sem in range(1, 11)}
    return sp.run_precheck(term, batches, studs, semesters, slots, courses, offers,
                           rooms, section_of=compute_section, **kwargs)


def codes(report, severity=None):
    return sorted(i.code for i in report.issues if severity in (None, i.severity))


# ── Tests ────────────────────────────────────────────────────────────────────

class TermConventionTests(unittest.TestCase):

    def test_even_term_offering_year_is_the_calendar_year_it_runs_in(self):
        self.assertEqual(sp.offering_year(2025, sp.ODD), 2025)
        self.assertEqual(sp.offering_year(2025, sp.EVEN), 2026)

    def test_semester_number_matches_get_user_semester(self):
        # get_user_semester: Odd -> 2*years + 1, Even -> 2*years (+1 year by
        # Jan-Jun); the dump's curr_semester_no agrees (batch 2025 in sem 2).
        self.assertEqual(sp.semester_number(2025, 2025, sp.ODD), 1)
        self.assertEqual(sp.semester_number(2025, 2025, sp.EVEN), 2)
        self.assertEqual(sp.semester_number(2025, 2024, sp.ODD), 3)
        self.assertEqual(sp.semester_number(2025, 2022, sp.EVEN), 8)

    def test_summer_term_is_rejected_not_mapped(self):
        report = run([batch(1)], students(1, 2), [], [], [],
                     term=sp.Term(2025, 'Summer Term'))
        self.assertTrue(report.blocking)
        self.assertEqual(codes(report), [sp.UNSUPPORTED_TERM])

    def test_constants_match_model_choices(self):
        def choices(path, model, attribute):
            for node in _parse(path).body:
                if isinstance(node, ast.ClassDef) and node.name == model:
                    for st in node.body:
                        if (isinstance(st, ast.Assign)
                                and getattr(st.targets[0], 'id', None) == attribute):
                            return [ast.literal_eval(p)[0] for p in st.value.elts]
            raise AssertionError(f'{model}.{attribute} not found')

        for path, model, attr in (
                ('programme_curriculum/models.py', 'CourseInstructor', 'SEMESTER_TYPE_CHOICES'),
                ('academic_procedures/models.py', 'TimetableVersion', 'SEMESTER_TYPE_CHOICES')):
            self.assertIn(sp.ODD, choices(path, model, attr))
            self.assertIn(sp.EVEN, choices(path, model, attr))

        slot_types = None
        for node in _parse('programme_curriculum/models.py').body:
            if (isinstance(node, ast.Assign)
                    and getattr(node.targets[0], 'id', None) == 'COURSESLOT_TYPE_CHOICES'):
                slot_types = {ast.literal_eval(p)[0] for p in node.value.elts}
        self.assertTrue(sp.NON_CORE_SLOT_TYPES <= slot_types)

    def test_ug_programmes_match_programme_scope(self):
        for node in ast.walk(_parse('globals/programme_scope.py')):
            if (isinstance(node, ast.Assign)
                    and getattr(node.targets[0], 'id', None) == 'PROGRAMME_NAMES_BY_SCOPE'):
                scope = ast.literal_eval(node.value)
        self.assertEqual(tuple(scope['UG']), sp.UG_PROGRAMMES)


class SectionMappingTests(unittest.TestCase):
    """Sections come from the real compute_section(); never re-implemented."""

    def test_compute_section_rules_used_by_precheck(self):
        self.assertEqual(compute_section(CSE, '25BCS001', 'B.Tech'), 'A')
        self.assertEqual(compute_section(CSE, '25BCS002', 'B.Tech'), 'B')
        self.assertEqual(compute_section(ECE, '25BEC001', 'B.Tech'), 'C')
        self.assertEqual(compute_section('Mechanical Engineering', '1', 'B.Tech'), 'D')
        self.assertEqual(compute_section('Design', '1', 'B.Des'), 'E')
        self.assertEqual(compute_section('Smart Manufacturing', '1', 'B.Tech'), 'F')
        self.assertIsNone(compute_section(CSE, '25MCS001', 'M.Tech'))

    def test_cse_batch_splits_by_roll_parity_with_real_counts(self):
        report = run([batch(1)], students(1, 5), [], [], [])
        sizes = {g.section: g.size for g in report.sections}
        self.assertEqual(sizes, {'A': 3, 'B': 2})
        self.assertTrue(all(g.semester_no == 2 for g in report.sections))

    def test_single_section_branches_and_pg_whole_batch(self):
        report = run([batch(1, discipline=ECE, acronym='ECE'),
                      batch(2, name='M.Tech AI & ML'),
                      batch(3, name='B.Des', discipline='Design', acronym='Des.')],
                     students(1, 4, prefix='25BEC')
                     + students(2, 3, programme='M.Tech', prefix='25MCS')
                     + students(3, 2, programme='B.Des', prefix='25BDS'),
                     [], [], [])
        got = {(g.batch_id, g.section): g.size for g in report.sections}
        self.assertEqual(got, {(1, 'C'): 4, (2, ''): 3, (3, 'E'): 2})

    def test_stored_section_wins_and_disagreement_is_a_warning(self):
        studs = students(1, 2, section='B')          # 001 would derive A
        report = run([batch(1)], studs, [], [], [])
        self.assertEqual({g.section: g.size for g in report.sections}, {'B': 2})
        self.assertIn(sp.SECTION_MISMATCH, codes(report, sp.WARNING))
        self.assertFalse(report.blocking)

    def test_underivable_ug_section_blocks(self):
        report = run([batch(1, discipline='Physics', acronym='PHY')], students(1, 2), [], [], [])
        self.assertIn(sp.SECTION_UNKNOWN, codes(report, sp.ERROR))
        self.assertEqual(report.sections, [])

    def test_cse_roll_without_digits_blocks(self):
        studs = [sp.StudentRecord('ABC', 'B.Tech', 1)]
        report = run([batch(1)], studs, [], [], [])
        self.assertIn(sp.SECTION_UNKNOWN, codes(report, sp.ERROR))

    def test_batch_edge_cases_are_warnings(self):
        report = run([batch(1), batch(2, year=2021, semesters=8), batch(3)],
                     students(1, 1) + students(2, 1) + [sp.StudentRecord('X1', 'B.Tech', None)],
                     [slot(1, 1, [])], [], [])
        warnings = codes(report, sp.WARNING)
        self.assertIn(sp.BATCH_PAST_FINAL_SEMESTER, warnings)      # 2021: sem 10 > 8
        self.assertIn(sp.BATCH_WITHOUT_STUDENTS, warnings)         # batch 3
        self.assertIn(sp.STUDENT_WITHOUT_BATCH, warnings)
        self.assertIn(sp.EMPTY_CORE_SLOT, warnings)
        self.assertFalse(report.blocking)

    def test_recorded_semester_disagreement_is_reported_but_still_counted(self):
        # 001 is on time (semester 2); 003 is recorded behind, like a repeater.
        studs = (students(1, 1, semester=2)
                 + [sp.StudentRecord('25BCS003', 'B.Tech', 1, None, 1)])
        report = run([batch(1)], studs, [slot(1, 1, [7])], [course(7)],
                     [offering(1, 7, 'f1', 'A')])
        self.assertEqual(codes(report, sp.WARNING), [sp.SEMESTER_MISMATCH])
        self.assertEqual(report.warnings[0].detail['recorded'], {2: 1, 1: 1})
        self.assertEqual([(g.section, g.size) for g in report.sections], [('A', 2)])
        self.assertEqual([c.size for c in report.classes], [2])
        self.assertFalse(report.blocking)

    def test_student_behind_stays_in_official_batch_not_junior_batch(self):
        # A CSE 2024 student recorded in semester 2 (the 2025 batch's semester)
        # is counted in 2024's section, and 2025's class size is unchanged.
        batches = [batch(1), batch(2, year=2024)]
        studs = (students(1, 1)                                         # 2025 A
                 + students(2, 1, prefix='24BCS', semester=4)           # on time
                 + [sp.StudentRecord('24BCS003', 'B.Tech', 2, None, 2)])  # behind
        slots = [slot(1, 1, [7]), slot(2, 2, [8], semester=4)]
        offers = [offering(1, 7, 'f1', 'A'), offering(2, 8, 'f2', 'A')]
        report = run(batches, studs, slots, [course(7), course(8)], offers)
        self.assertFalse(report.blocking, report.as_text())
        sizes = {(g.year, g.section): g.size for g in report.sections}
        self.assertEqual(sizes, {(2025, 'A'): 1, (2024, 'A'): 2})
        self.assertEqual(sorted((c.course_code, c.size) for c in report.classes),
                         [('C7', 1), ('C8', 2)])
        self.assertEqual(codes(report, sp.WARNING), [sp.SEMESTER_MISMATCH])
        self.assertIn('still counted', report.warnings[0].message)


class WeeklyHoursTests(unittest.TestCase):

    def test_weekly_patterns_consistent_with_credits_are_accepted(self):
        hours, _ = sp.hours_from_course(course(1, lec=3, tut=1, prac=0, credit=4))
        self.assertEqual((hours.lectures, hours.tutorials, hours.practical_blocks), (3, 1, 0))
        hours, _ = sp.hours_from_course(course(1, lec=3, tut=0, prac=2, credit=4))
        self.assertEqual((hours.practical_blocks, hours.practical_block_size), (1, 2))

    def test_untrustworthy_values_are_rejected_not_guessed(self):
        cases = [
            course(1, lec=30, tut=10, prac=20, credit=4),    # semester totals
            course(1, lec=None, tut=None, prac=None, credit=3),
            course(1, lec=0, tut=0, prac=0, credit=2),
            course(1, lec=3, tut=0, prac=0, credit=4),       # credit mismatch
            course(1, lec=2, tut=0, prac=4, credit=4),       # 4h lab: block split unknown
        ]
        for c in cases:
            with self.subTest(c=c):
                hours, why = sp.hours_from_course(c)
                self.assertIsNone(hours)
                self.assertTrue(why)

    def test_unknown_hours_block_generation(self):
        report = run([batch(1)], students(1, 2), [slot(1, 1, [7])],
                     [course(7, lec=30, tut=0, prac=20, credit=4)],
                     [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')])
        self.assertEqual(codes(report, sp.ERROR), [sp.HOURS_UNKNOWN])
        self.assertEqual(report.classes, [])

    def test_explicit_weekly_requirement_overrides_course_values(self):
        explicit = sp.WeeklyHours(lectures=2, tutorials=0, practical_blocks=2,
                                  practical_block_size=2, preferred_room_type='Computer Lab')
        report = run([batch(1)], students(1, 2), [slot(1, 1, [7])],
                     [course(7, lec=30, tut=0, prac=20, credit=4)],
                     [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')],
                     weekly_hours={7: explicit})
        self.assertFalse(report.blocking)
        self.assertEqual({c.hours_source for c in report.classes}, {'requirement'})


class OthersSlotTests(unittest.TestCase):
    """A single-course 'Others' slot is a required class; other types unchanged."""

    def setUp(self):
        self.batches = [batch(1, discipline=ECE, acronym='ECE')]     # one section, C
        self.students = students(1, 3, prefix='25BEC')

    def check(self, slots, courses, offers, rooms=ROOMS):
        return run(self.batches, self.students, slots, courses, offers, rooms=rooms)

    def test_others_is_not_a_skipped_type(self):
        self.assertNotIn('Others', sp.NON_CORE_SLOT_TYPES)

    def test_valid_single_course_others_slot_becomes_a_class(self):
        report = self.check([slot(1, 1, [7], slot_type='Others')], [course(7)],
                            [offering(1, 7, 'f1')])
        self.assertEqual(report.issues, [])
        [planned] = report.classes
        self.assertEqual((planned.course_code, planned.section, planned.size,
                          planned.instructor_id), ('C7', 'C', 3, 'f1'))

    def test_others_slot_without_offering_blocks(self):
        report = self.check([slot(1, 1, [7], slot_type='Others')], [course(7)], [])
        self.assertEqual(codes(report), [sp.MISSING_OFFERING])

    def test_others_slot_with_unknown_hours_blocks(self):
        report = self.check([slot(1, 1, [7], slot_type='Others')],
                            [course(7, lec=10, tut=0, prac=0, credit=1)],   # PC1002-like
                            [offering(1, 7, 'f1')])
        self.assertEqual(codes(report), [sp.HOURS_UNKNOWN])
        self.assertEqual(report.classes, [])

    def test_others_slot_room_checks_apply(self):
        too_small = self.check([slot(1, 1, [7], slot_type='Others')], [course(7)],
                               [offering(1, 7, 'f1')], rooms=[hall('LH', 3)])
        self.assertEqual(sorted(set(codes(too_small))), [sp.NO_ROOM_FITS])  # needs 4 seats
        no_rooms = self.check([slot(1, 1, [7], slot_type='Others')], [course(7)],
                              [offering(1, 7, 'f1')], rooms=[])
        self.assertEqual(codes(no_rooms), [sp.NO_ROOMS])

    def test_multi_course_and_empty_others_slots(self):
        report = self.check([slot(1, 1, [7, 8], slot_type='Others'), slot(2, 1, [], slot_type='Others')],
                            [course(7), course(8)], [])
        self.assertFalse(report.blocking)
        self.assertEqual(codes(report), [sp.EMPTY_CORE_SLOT, sp.NON_CORE_SKIPPED])
        skipped = next(i for i in report.issues if i.code == sp.NON_CORE_SKIPPED)
        self.assertIn('choice of 2 courses', skipped.message)

    def test_other_single_course_types_are_still_skipped(self):
        for slot_type in sorted(sp.NON_CORE_SLOT_TYPES):
            with self.subTest(slot_type=slot_type):
                report = self.check([slot(1, 1, [7], slot_type=slot_type)], [course(7)], [])
                self.assertEqual(codes(report), [sp.NON_CORE_SKIPPED])
                self.assertEqual(report.classes, [])

    def test_course_required_in_one_batch_and_skipped_in_another_is_reported(self):
        batches = self.batches + [batch(2, name='B.Tech', discipline='Smart Manufacturing',
                                        acronym='SM')]
        studs = self.students + students(2, 2, prefix='25BSM')
        report = run(batches, studs,
                     [slot(1, 1, [7], slot_type='Others'), slot(2, 2, [7], slot_type='Optional')],
                     [course(7)], [offering(1, 7, 'f1')])
        self.assertFalse(report.blocking)
        self.assertEqual(codes(report), [sp.NON_CORE_SKIPPED, sp.SLOT_TYPE_INCONSISTENT])
        issue = next(i for i in report.issues if i.code == sp.SLOT_TYPE_INCONSISTENT)
        self.assertEqual(issue.detail, {'required_in': ['B.Tech ECE 2025'],
                                        'skipped_in': ['B.Tech SM 2025 (Optional)']})
        self.assertEqual([(c.batch_label, c.section) for c in report.classes],
                         [('B.Tech ECE 2025', 'C')])


class OfferingResolutionTests(unittest.TestCase):

    def test_labelled_offering_per_section(self):
        chosen, why = sp.resolve_section_offerings(
            ['A', 'B'], [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')])
        self.assertEqual(why, '')
        self.assertEqual({s: o.instructor_id for s, o in chosen.items()},
                         {'A': 'f1', 'B': 'f2'})

    def test_one_unlabelled_offering_for_one_section(self):
        chosen, why = sp.resolve_section_offerings(['C'], [offering(1, 7, 'f1')])
        self.assertEqual((why, chosen['C'].instructor_id), ('', 'f1'))

    def test_one_unlabelled_offering_shared_by_sections_is_ambiguous(self):
        chosen, why = sp.resolve_section_offerings(['A', 'B'], [offering(1, 7, 'f1')])
        self.assertEqual(chosen, {})
        self.assertIn('combined class or separate classes', why)
        self.assertIn('unassigned: A, B', why)

    def test_several_unlabelled_offerings_are_ambiguous(self):
        chosen, why = sp.resolve_section_offerings(
            ['D'], [offering(1, 7, 'f1'), offering(2, 7, 'f2')])
        self.assertEqual(chosen, {})
        self.assertIn('2 offerings without a section label', why)

    def test_lone_offering_for_another_section_is_not_borrowed(self):
        chosen, why = sp.resolve_section_offerings(['E'], [offering(1, 7, 'f1', 'A')])
        self.assertEqual(chosen, {})
        self.assertIn('labelled A only', why)

    def test_partial_labels_resolve_what_they_can(self):
        chosen, why = sp.resolve_section_offerings(
            ['A', 'B'], [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2')])
        self.assertEqual(list(chosen), ['A'])
        self.assertIn('unassigned: B', why)

    def test_agrees_with_pick_offering_whenever_precheck_accepts(self):
        cases = [
            (['A', 'B'], [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')]),
            (['C'], [offering(1, 7, 'f1')]),
            (['A', 'B'], [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B'),
                          offering(3, 7, 'f3')]),
        ]
        for sections, offers in cases:
            chosen, _ = sp.resolve_section_offerings(sections, offers)
            for section, picked in chosen.items():
                with self.subTest(section=section):
                    expected = pick_offering(SimpleNamespace(section=section), offers)
                    self.assertIs(picked, expected)


class PrecheckReportTests(unittest.TestCase):

    def setUp(self):
        # CSE 2025 (A: 3, B: 2) and ECE 2025 (C: 2), all in semester 2.
        self.batches = [batch(1), batch(2, discipline=ECE, acronym='ECE')]
        self.students = students(1, 5) + students(2, 2, prefix='25BEC')

    def test_clean_data_is_ready_with_one_class_per_section(self):
        # Slots 4-5 are not core: a choice of courses, and a single-course
        # 'Project' slot (course 11 has no offering, so scheduling it would fail).
        slots = [slot(1, 1, [7]), slot(2, 2, [7]), slot(3, 1, [8]),
                 slot(4, 1, [9, 10], slot_type='Open Elective'),
                 slot(5, 2, [11], slot_type='Project')]
        offers = [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B'),
                  offering(3, 7, 'f1', 'C'), offering(4, 8, 'f3', 'A'),
                  offering(5, 8, 'f3', 'B')]
        report = run(self.batches, self.students, slots,
                     [course(7), course(8, lec=3, tut=0, prac=2), course(11)], offers)
        self.assertFalse(report.blocking, report.as_text())
        got = sorted((c.course_code, c.section, c.size, c.instructor_id) for c in report.classes)
        self.assertEqual(got, [('C7', 'A', 3, 'f1'), ('C7', 'B', 2, 'f2'),
                               ('C7', 'C', 2, 'f1'),
                               ('C8', 'A', 3, 'f3'), ('C8', 'B', 2, 'f3')])
        self.assertEqual(codes(report, sp.WARNING), [sp.NON_CORE_SKIPPED] * 2)

    def test_unlabelled_offering_for_several_sections_blocks_and_names_them(self):
        report = run(self.batches, self.students, [slot(1, 1, [7]), slot(2, 2, [7])],
                     [course(7)], [offering(1, 7, 'f1')])
        self.assertEqual(codes(report, sp.ERROR), [sp.AMBIGUOUS_OFFERING])
        issue = report.errors[0]
        self.assertEqual(issue.detail['sections'],
                         ['B.Tech CSE 2025 A', 'B.Tech CSE 2025 B', 'B.Tech ECE 2025 C'])
        self.assertEqual(report.classes, [])

    def test_missing_offering_blocks(self):
        report = run(self.batches, self.students, [slot(1, 2, [7])], [course(7)], [])
        self.assertEqual(codes(report, sp.ERROR), [sp.MISSING_OFFERING])

    def test_offering_from_another_term_is_not_used(self):
        wrong_year = offering(1, 7, 'f1', year=2025)                 # = 2025-26 Odd
        wrong_type = offering(2, 7, 'f2', semester_type=sp.ODD)
        report = run(self.batches, self.students, [slot(1, 2, [7])], [course(7)],
                     [wrong_year, wrong_type])
        self.assertEqual(codes(report, sp.ERROR), [sp.MISSING_OFFERING])
        report = run(self.batches, self.students, [slot(1, 2, [7])], [course(7)],
                     [offering(1, 7, 'f1', year=2026)])
        self.assertFalse(report.blocking)

    def test_room_capacity_uses_section_size_with_buffer(self):
        big = [batch(1)]
        studs = students(1, 304)                       # A: 152, B: 152
        offers = [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')]
        self.assertEqual(sp.seats_needed(152), 168)
        short = run(big, studs, [slot(1, 1, [7])], [course(7)], offers, rooms=[hall('LH', 167)])
        self.assertEqual(set(codes(short, sp.ERROR)), {sp.NO_ROOM_FITS})
        self.assertIn('needs 168 seats', short.errors[0].message)
        enough = run(big, studs, [slot(1, 1, [7])], [course(7)], offers, rooms=[hall('LH', 168)])
        self.assertFalse(enough.blocking)            # the 304-student batch never needs one room
        self.assertEqual({c.seats_needed for c in enough.classes}, {168})

    def test_inactive_and_wrong_type_rooms_do_not_count(self):
        offers = [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')]
        rooms = [hall('LH', 200), hall('LAB', 200, 'Computer Lab', active=False)]
        report = run(self.batches[:1], self.students[:5], [slot(1, 1, [7])],
                     [course(7, lec=3, tut=0, prac=2)], offers, rooms=rooms)
        self.assertEqual(set(codes(report, sp.ERROR)), {sp.NO_ROOM_FITS})
        self.assertTrue(all(i.detail['slot_type'] == sp.PRACTICAL for i in report.errors))

    def test_no_rooms_blocks(self):
        report = run(self.batches, self.students, [], [], [], rooms=[])
        self.assertEqual(codes(report, sp.ERROR), [sp.NO_ROOMS])

    def test_instructor_hours_beyond_the_week_block(self):
        # f1 owns 3 sections x 3 courses x 5 hours = 45 > 40 slots.
        slots = [slot(1, 1, [7, ]), slot(2, 1, [8]), slot(3, 1, [9]),
                 slot(4, 2, [7]), slot(5, 2, [8]), slot(6, 2, [9])]
        courses = [course(c, lec=4, tut=1, credit=5) for c in (7, 8, 9)]
        offers = [offering(10 * c + i, c, 'f1', s)
                  for c in (7, 8, 9) for i, s in enumerate('ABC')]
        report = run(self.batches, self.students, slots, courses, offers)
        self.assertEqual(codes(report, sp.ERROR), [sp.INSTRUCTOR_OVERLOADED])

    def test_section_hours_beyond_the_week_block(self):
        slots = [slot(n, 2, [n]) for n in range(1, 10)]
        courses = [course(n, lec=4, tut=1, credit=5) for n in range(1, 10)]   # 45 h
        offers = [offering(n, n, f'f{n}') for n in range(1, 10)]
        report = run(self.batches, self.students, slots, courses, offers)
        self.assertEqual(codes(report, sp.ERROR), [sp.SECTION_OVERLOADED])

    def test_pg_batches_sharing_a_course_cannot_be_told_apart(self):
        batches = [batch(1, name='M.Tech AI & ML'), batch(2, name='M.Tech Data Science')]
        studs = (students(1, 2, programme='M.Tech', prefix='25MCS')
                 + students(2, 2, programme='M.Tech', prefix='25MDS'))
        report = run(batches, studs, [slot(1, 1, [7]), slot(2, 2, [7])], [course(7)],
                     [offering(1, 7, 'f1')])
        self.assertEqual(codes(report, sp.ERROR), [sp.AMBIGUOUS_OFFERING])

    def test_same_label_in_two_batches_is_not_shared(self):
        # CSE 2024 and CSE 2025 both have sections A and B; labels alone do
        # not say which batch an 'A' offering is for.
        batches = [batch(1), batch(2, year=2024)]
        studs = students(1, 2) + students(2, 2, prefix='24BCS')
        slots = [slot(1, 1, [7]), slot(2, 2, [7], semester=4)]
        offers = [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')]
        report = run(batches, studs, slots, [course(7)], offers)
        self.assertEqual(codes(report, sp.ERROR), [sp.AMBIGUOUS_OFFERING])
        self.assertEqual(report.classes, [])
        none = run(batches, studs, slots, [course(7)], [])
        self.assertEqual(codes(none, sp.ERROR), [sp.MISSING_OFFERING])
        self.assertIn('no offering for this term', none.errors[0].message)
        self.assertEqual(none.classes, [])

    def test_course_in_two_required_slots_of_one_batch_counts_each_section_once(self):
        # Course 7 is listed in two required slots of CSE 2025 (e.g. a core
        # slot and an 'Others' slot).  22 hours a week: counted twice it
        # would also exceed the 40-hour week.
        slots = [slot(1, 1, [7]), slot(2, 1, [7], slot_type='Others')]
        courses = [course(7, lec=22, tut=0, credit=22)]
        labelled = [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B')]
        report = run(self.batches[:1], self.students[:5], slots, courses, labelled)
        self.assertEqual(report.issues, [])
        self.assertEqual(sorted((c.section, c.size, c.instructor_id) for c in report.classes),
                         [('A', 3, 'f1'), ('B', 2, 'f2')])

        # Genuine ambiguity is still reported: one unlabelled offering for A and B.
        shared = run(self.batches[:1], self.students[:5], slots, courses, [offering(1, 7, 'f1')])
        self.assertEqual(codes(shared), [sp.AMBIGUOUS_OFFERING])
        self.assertEqual(shared.errors[0].detail['sections'],
                         ['B.Tech CSE 2025 A', 'B.Tech CSE 2025 B'])
        self.assertIn('shared by 2 sections', shared.errors[0].message)

    def test_structural_gaps_block(self):
        no_curriculum = sp.BatchRecord(1, 'B.Tech', CSE, 'CSE', 2025, None, 8)
        report = run([no_curriculum, batch(2), batch(3)],
                     students(1, 1) + students(2, 1) + students(3, 1),
                     [], [], [], semesters={(3, 2)})
        self.assertEqual(codes(report, sp.ERROR), [sp.NO_CURRICULUM, sp.NO_SEMESTER])
        # Batch 3's semester exists but has no course slots: reported, not blocking.
        self.assertEqual(codes(report, sp.WARNING), [sp.EMPTY_SEMESTER])

    def test_offering_for_a_section_nobody_is_in_is_reported(self):
        offers = [offering(1, 7, 'f1', 'A'), offering(2, 7, 'f2', 'B'),
                  offering(3, 7, 'f3', 'G')]
        report = run(self.batches[:1], self.students[:5], [slot(1, 1, [7])], [course(7)], offers)
        self.assertFalse(report.blocking)
        self.assertEqual(codes(report, sp.WARNING), [sp.UNUSED_OFFERING])

    def test_every_issue_severity_matches_its_code(self):
        report = run(self.batches, self.students + [sp.StudentRecord('Z', 'B.Tech', None)],
                     [slot(1, 1, [7]), slot(2, 2, [8]), slot(3, 1, [9], 'Swayam')],
                     [course(7), course(8, lec=30, tut=0, prac=0, credit=3)],
                     [offering(1, 7, 'f1')], rooms=[])
        self.assertTrue(report.issues)
        for issue in report.issues:
            with self.subTest(code=issue.code):
                self.assertEqual(issue.severity == sp.ERROR, issue.code in sp.ERROR_CODES)

    def test_report_serialises_and_summarises(self):
        report = run(self.batches, self.students, [slot(1, 1, [7])], [course(7)],
                     [offering(1, 7, 'f1')])
        data = json.loads(json.dumps(report.as_dict()))
        self.assertTrue(data['blocking'])
        self.assertEqual(data['summary']['students'], 7)
        self.assertEqual(data['summary']['by_code'],
                         {sp.AMBIGUOUS_OFFERING: 1, sp.EMPTY_SEMESTER: 1})   # ECE has no slots
        text = report.as_text()
        self.assertIn('Result: BLOCKED', text)
        self.assertIn('ERRORS (block generation)', text)
        self.assertIn('AMBIGUOUS_OFFERING (1)', text)
        self.assertIn('WARNINGS', text)
        self.assertLess(text.index('ERRORS'), text.index('WARNINGS'))


if __name__ == '__main__':
    unittest.main()
