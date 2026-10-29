"""
Section-wise timetable pre-check
================================
Checks, before any solving, whether every core course of every section in a
term can be scheduled from the data as it stands, and says exactly what is
missing when it cannot.  Nothing is guessed: an instructor, a weekly-hour
pattern or a room is only accepted when the data states it unambiguously.

Data model (programme_curriculum):

    running Batch -> Curriculum -> Semester(semester_no) -> CourseSlot -> Course
    CourseInstructor(course, year, semester_type, section_label) = one offering

A batch is split into sections by ``Student.section`` or, when that is empty,
by ``compute_section()``; each core course becomes one class per section.

The module is pure Python (no Django, no OR-Tools).  Callers pass plain
records and the section function, so the same code serves the future DB
loader and the tests::

    from applications.academic_information.models import compute_section
    report = run_precheck(term, batches, students, semesters, slots, courses,
                          offerings, rooms, section_of=compute_section)
    report.blocking      # True when generation must not start
    report.as_dict()     # JSON-ready
    report.as_text()     # human-readable summary

Every issue is either an ERROR (blocks generation) or a WARNING (reported,
does not block).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from applications.academic_procedures.solver.timetable_solver import (
    CAPACITY_BUFFER,
    DAYS,
    HOURS,
    LECTURE,
    PRACTICAL,
    TUTORIAL,
    eligible_room_types,
)

ERROR = 'ERROR'
WARNING = 'WARNING'

ODD, EVEN = 'Odd Semester', 'Even Semester'

# Must match PROGRAMME_NAMES_BY_SCOPE['UG'] in globals/programme_scope.py,
# which cannot be imported without Django (checked by a test).
UG_PROGRAMMES = ('B.Tech', 'B.Des')

# CourseSlot types that are a choice for the student, not a class the whole
# section attends.  Slots of these types are reported and not scheduled.
# 'Others' is not listed: a single-course 'Others' slot (e.g. a Professional
# Development Course) is a required class; with several courses it is a
# choice and is skipped like any multi-course slot.
NON_CORE_SLOT_TYPES = frozenset({
    'Professional Elective', 'Open Elective', 'Swayam', 'Optional',
    'Backlog', 'Project',
})

# Practical hours accepted from Course.pratical_hours: one lab block of that
# length.  Larger values need an explicit weekly requirement, because how they
# split into blocks (2+2, 3+3, 2+2+2 ...) is a decision, not data.
SINGLE_BLOCK_PRACTICAL_HOURS = (2, 3)

# Teaching hours available to one person or one section in a week.
WEEKLY_TEACHING_HOURS = len(DAYS) * len(HOURS)

# Issue codes.  ERROR codes block generation.
UNSUPPORTED_TERM = 'UNSUPPORTED_TERM'
NO_CURRICULUM = 'NO_CURRICULUM'
NO_SEMESTER = 'NO_SEMESTER'
SECTION_UNKNOWN = 'SECTION_UNKNOWN'
MISSING_OFFERING = 'MISSING_OFFERING'
AMBIGUOUS_OFFERING = 'AMBIGUOUS_OFFERING'
HOURS_UNKNOWN = 'HOURS_UNKNOWN'
NO_ROOMS = 'NO_ROOMS'
NO_ROOM_FITS = 'NO_ROOM_FITS'
INSTRUCTOR_OVERLOADED = 'INSTRUCTOR_OVERLOADED'
SECTION_OVERLOADED = 'SECTION_OVERLOADED'
# WARNING codes.
SECTION_MISMATCH = 'SECTION_MISMATCH'
SEMESTER_MISMATCH = 'SEMESTER_MISMATCH'
STUDENT_WITHOUT_BATCH = 'STUDENT_WITHOUT_BATCH'
BATCH_WITHOUT_STUDENTS = 'BATCH_WITHOUT_STUDENTS'
BATCH_PAST_FINAL_SEMESTER = 'BATCH_PAST_FINAL_SEMESTER'
NON_CORE_SKIPPED = 'NON_CORE_SKIPPED'
SLOT_TYPE_INCONSISTENT = 'SLOT_TYPE_INCONSISTENT'
EMPTY_SEMESTER = 'EMPTY_SEMESTER'
EMPTY_CORE_SLOT = 'EMPTY_CORE_SLOT'
UNUSED_OFFERING = 'UNUSED_OFFERING'

ERROR_CODES = frozenset({
    UNSUPPORTED_TERM, NO_CURRICULUM, NO_SEMESTER, SECTION_UNKNOWN,
    MISSING_OFFERING, AMBIGUOUS_OFFERING, HOURS_UNKNOWN, NO_ROOMS,
    NO_ROOM_FITS, INSTRUCTOR_OVERLOADED, SECTION_OVERLOADED,
})


# ── Input records ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Term:
    academic_year: int       # start year, as TimetableVersion.academic_year
    semester_type: str       # TimetableVersion.semester_type


@dataclass(frozen=True)
class BatchRecord:
    batch_id: int
    name: str                # Batch.name, e.g. 'B.Tech', 'M.Tech AI & ML'
    discipline_name: str     # Discipline.name, as compute_section expects
    discipline_acronym: str
    year: int                # admission year
    curriculum_id: Optional[int]
    no_of_semester: Optional[int] = None

    @property
    def label(self) -> str:
        return f'{self.name} {self.discipline_acronym} {self.year}'


@dataclass(frozen=True)
class StudentRecord:
    roll_no: str
    programme: str           # Student.programme
    batch_id: Optional[int]
    section: Optional[str] = None          # Student.section as stored
    curr_semester_no: Optional[int] = None


@dataclass(frozen=True)
class CourseSlotRecord:
    slot_id: int
    curriculum_id: int
    semester_no: int
    name: str
    slot_type: str
    course_ids: Tuple[int, ...]


@dataclass(frozen=True)
class CourseRecord:
    course_id: int
    code: str
    name: str
    credit: Optional[int]
    lecture_hours: Optional[int]
    tutorial_hours: Optional[int]
    practical_hours: Optional[int]


@dataclass(frozen=True)
class OfferingRecord:
    offering_id: int
    course_id: int
    instructor_id: str       # Faculty pk (= ExtraInfo id): the person
    year: int
    semester_type: str
    section_label: Optional[str]


@dataclass(frozen=True)
class WeeklyHours:
    """Explicit weekly pattern; overrides Course L/T/P when given."""
    lectures: int
    tutorials: int
    practical_blocks: int = 0
    practical_block_size: int = 0
    preferred_room_type: str = 'any'

    @property
    def total_hours(self) -> int:
        return (self.lectures + self.tutorials
                + self.practical_blocks * self.practical_block_size)


@dataclass(frozen=True)
class RoomRecord:
    room_id: str
    capacity: int
    room_type: str
    is_active: bool = True


# ── Output records ───────────────────────────────────────────────────────────

@dataclass
class Issue:
    severity: str
    code: str
    message: str
    batch: str = ''
    section: str = ''
    course_code: str = ''
    detail: Dict = field(default_factory=dict)


@dataclass
class SectionGroup:
    batch_id: int
    batch_label: str
    programme: str
    discipline: str
    year: int
    semester_no: int
    section: str             # '' = the whole batch (PG, no sections)
    size: int


@dataclass
class PlannedClass:
    """One course taught to one section, ready for the solver."""
    batch_id: int
    batch_label: str
    section: str
    course_id: int
    course_code: str
    size: int
    seats_needed: int
    offering_id: int
    instructor_id: str
    hours: WeeklyHours
    hours_source: str        # 'requirement' | 'course L/T/P'


@dataclass
class PrecheckReport:
    term: Term
    offering_year: Optional[int]
    sections: List[SectionGroup] = field(default_factory=list)
    classes: List[PlannedClass] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == WARNING]

    @property
    def blocking(self) -> bool:
        return bool(self.errors)

    def counts(self) -> Dict[str, int]:
        return dict(Counter(i.code for i in self.issues))

    def as_dict(self) -> Dict:
        return {
            'term': asdict(self.term),
            'offering_year': self.offering_year,
            'blocking': self.blocking,
            'summary': {
                'sections': len(self.sections),
                'students': sum(s.size for s in self.sections),
                'classes_ready': len(self.classes),
                'errors': len(self.errors),
                'warnings': len(self.warnings),
                'by_code': self.counts(),
            },
            'sections': [asdict(s) for s in self.sections],
            'classes': [asdict(c) for c in self.classes],
            'errors': [asdict(i) for i in self.errors],
            'warnings': [asdict(i) for i in self.warnings],
        }

    def as_text(self) -> str:
        t = self.term
        lines = [
            f'Section-wise pre-check: {t.academic_year}-{(t.academic_year + 1) % 100:02d} '
            f'{t.semester_type} (offering year {self.offering_year})',
            f'Result: {"BLOCKED" if self.blocking else "READY"} - '
            f'{len(self.errors)} error(s), {len(self.warnings)} warning(s); '
            f'{len(self.sections)} section(s), '
            f'{sum(s.size for s in self.sections)} student(s), '
            f'{len(self.classes)} class(es) ready.',
        ]
        for title, issues in (('ERRORS (block generation)', self.errors),
                              ('WARNINGS', self.warnings)):
            if not issues:
                continue
            lines.append('')
            lines.append(title)
            by_code = defaultdict(list)
            for issue in issues:
                by_code[issue.code].append(issue)
            for code in sorted(by_code):
                lines.append(f'  {code} ({len(by_code[code])})')
                for issue in by_code[code]:
                    lines.append(f'    - {issue.message}')
        return '\n'.join(lines)


# ── Term conventions ─────────────────────────────────────────────────────────

def offering_year(academic_year: int, semester_type: str) -> int:
    """CourseInstructor.year for a term.

    TimetableVersion stores the start year of the academic year; the offering
    stores the calendar year the term runs in, so an Even term of 2025-26 is
    CourseInstructor.year 2026 (see CourseInstructor.academic_year).
    """
    return academic_year + 1 if semester_type == EVEN else academic_year


def semester_number(academic_year: int, admission_year: int, semester_type: str) -> int:
    """Semester a batch is in, as get_user_semester() counts it.

    Odd terms are 1, 3, 5 ...; Even terms 2, 4, 6 ... of the batch.
    """
    base = 2 * (academic_year - admission_year)
    return base + 1 if semester_type == ODD else base + 2


# ── Weekly hours ─────────────────────────────────────────────────────────────

def hours_from_course(course: CourseRecord) -> Tuple[Optional[WeeklyHours], str]:
    """Weekly pattern from Course L/T/P, or (None, reason) when not trustworthy.

    Course.lecture/tutorial/pratical_hours mix weekly values (3-1-0) with
    semester totals (30-10-20) and blanks, so they are accepted only when they
    agree with the credits under the L + T + P/2 rule and the practical hours
    form a single 2- or 3-hour lab block.
    """
    values = (course.lecture_hours, course.tutorial_hours, course.practical_hours)
    if course.credit is None or any(v is None for v in values):
        return None, 'L/T/P or credit is blank'
    lec, tut, prac = values
    if lec + tut + prac == 0:
        return None, 'L/T/P are all 0'
    if course.credit != lec + tut + prac / 2:
        return None, (f'L/T/P {lec}-{tut}-{prac} do not match {course.credit} '
                      f'credits (expected L + T + P/2); they may be semester totals')
    if prac and prac not in SINGLE_BLOCK_PRACTICAL_HOURS:
        return None, (f'{prac} practical hours do not form one 2- or 3-hour '
                      f'lab block; the block split needs a weekly requirement')
    return WeeklyHours(
        lectures=lec, tutorials=tut,
        practical_blocks=1 if prac else 0, practical_block_size=prac,
    ), ''


# ── Sections ─────────────────────────────────────────────────────────────────

def student_section(student: StudentRecord, batch: BatchRecord,
                    section_of: Callable) -> Tuple[Optional[str], Optional[str]]:
    """(section, derived) for one student.

    ``section`` is '' for programmes without sections and None when a UG
    student's section cannot be determined.  The stored Student.section wins;
    ``derived`` is what ``section_of`` (compute_section) gives, so callers can
    report a disagreement.
    """
    derived = section_of(batch.discipline_name, student.roll_no, student.programme)
    stored = (student.section or '').strip() or None
    if student.programme not in UG_PROGRAMMES:
        return stored or '', derived
    return stored or derived, derived


def build_sections(term: Term, batches: Iterable[BatchRecord],
                   students: Iterable[StudentRecord], section_of: Callable,
                   issues: List[Issue]) -> List[SectionGroup]:
    batch_by_id = {b.batch_id: b for b in batches}
    members: Dict[int, List[StudentRecord]] = defaultdict(list)
    no_batch = 0
    for stu in students:
        if stu.batch_id is None:
            no_batch += 1
        elif stu.batch_id in batch_by_id:
            members[stu.batch_id].append(stu)
    if no_batch:
        issues.append(Issue(WARNING, STUDENT_WITHOUT_BATCH,
                            f'{no_batch} student(s) have no batch and are not scheduled.',
                            detail={'count': no_batch}))

    groups: List[SectionGroup] = []
    for batch in sorted(batch_by_id.values(), key=lambda b: (b.name, b.discipline_acronym, b.year)):
        sem_no = semester_number(term.academic_year, batch.year, term.semester_type)
        if sem_no < 1:
            continue
        if batch.no_of_semester and sem_no > batch.no_of_semester:
            issues.append(Issue(
                WARNING, BATCH_PAST_FINAL_SEMESTER,
                f'{batch.label}: semester {sem_no} is past the curriculum\'s '
                f'{batch.no_of_semester} semesters; not scheduled.',
                batch=batch.label, detail={'semester_no': sem_no}))
            continue
        studs = members.get(batch.batch_id, [])
        if not studs:
            issues.append(Issue(WARNING, BATCH_WITHOUT_STUDENTS,
                                f'{batch.label}: running batch with no students; not scheduled.',
                                batch=batch.label))
            continue

        sizes: Counter = Counter()
        unknown, mismatched = [], []
        stored_sems: Counter = Counter()
        for stu in studs:
            section, derived = student_section(stu, batch, section_of)
            if section is None:
                unknown.append(stu.roll_no)
                continue
            if stu.section and derived and stu.section.strip() != derived:
                mismatched.append(stu.roll_no)
            sizes[section] += 1
            if stu.curr_semester_no is not None:
                stored_sems[stu.curr_semester_no] += 1
        if unknown:
            issues.append(Issue(
                ERROR, SECTION_UNKNOWN,
                f'{batch.label}: section cannot be determined for {len(unknown)} '
                f'student(s) (e.g. {", ".join(sorted(unknown)[:3])}); class sizes would be wrong.',
                batch=batch.label, detail={'roll_nos': sorted(unknown)}))
        if mismatched:
            issues.append(Issue(
                WARNING, SECTION_MISMATCH,
                f'{batch.label}: stored section differs from compute_section() for '
                f'{len(mismatched)} student(s); the stored section is used.',
                batch=batch.label, detail={'roll_nos': sorted(mismatched)}))
        # Students behind (e.g. repeaters) stay in their official batch: the
        # data does not say which junior classes, if any, they attend.
        off = {n: c for n, c in stored_sems.items() if n != sem_no}
        if off:
            issues.append(Issue(
                WARNING, SEMESTER_MISMATCH,
                f'{batch.label}: term is semester {sem_no} but '
                f'{sum(off.values())} student(s) are recorded in semester(s) '
                f'{sorted(off)}; they are still counted in this batch\'s sections. '
                f'Check curr_semester_no if this is the current term.',
                batch=batch.label, detail={'expected': sem_no, 'recorded': dict(stored_sems)}))
        for section in sorted(sizes):
            groups.append(SectionGroup(
                batch_id=batch.batch_id, batch_label=batch.label,
                programme=batch.name, discipline=batch.discipline_acronym,
                year=batch.year, semester_no=sem_no, section=section,
                size=sizes[section]))
    return groups


# ── Instructors ──────────────────────────────────────────────────────────────

def resolve_section_offerings(sections: List[str], offerings: List[OfferingRecord]
                              ) -> Tuple[Dict[str, OfferingRecord], str]:
    """Offering for each section that needs the course, plus why any are left.

    Accepted without guessing:
      * the offering labelled with the section; or
      * the only offering, unlabelled, when exactly one section needs the course.

    Everything else is left unassigned with a reason.  In particular an
    unlabelled offering shared by several sections is rejected: the data does
    not say whether they form one combined class or separate classes, nor who
    teaches each section.  (pick_offering() would put every student in that
    one offering, i.e. one combined class.)  This is stricter than
    pick_offering(), which also accepts a lone offering labelled for another
    section.
    """
    if not offerings:
        return {}, 'no offering for this term'
    labelled = {o.section_label: o for o in offerings if o.section_label}
    unlabelled = [o for o in offerings if not o.section_label]
    chosen = {s: labelled[s] for s in sections if s in labelled}
    rest = [s for s in sections if s not in chosen]
    if not rest:
        return chosen, ''
    if len(sections) == 1 and len(offerings) == 1 and unlabelled:
        return {sections[0]: unlabelled[0]}, ''
    who = sorted({o.instructor_id for o in unlabelled})
    if len(unlabelled) > 1:
        reason = (f'{len(unlabelled)} offerings without a section label '
                  f'({", ".join(who)}); which one teaches which section is not recorded')
    elif unlabelled and len(sections) == 1:
        reason = (f'an offering without a section label ({who[0]}) exists beside '
                  f'offerings for sections {", ".join(sorted(labelled))}; which one '
                  f'this section attends is not recorded')
    elif unlabelled:
        reason = (f'an offering without a section label ({who[0]}) is shared by '
                  f'{len(sections)} sections; whether they are one combined class or '
                  f'separate classes, and who teaches each, is not recorded')
    else:
        reason = (f'offerings are labelled {", ".join(sorted(labelled))} only')
    return chosen, f'{reason} [unassigned: {", ".join(s or "(whole batch)" for s in rest)}]'


# ── Rooms ────────────────────────────────────────────────────────────────────

def seats_needed(size: int, buffer: float = CAPACITY_BUFFER) -> int:
    return math.ceil(size * buffer)


def room_shortfall(slot_type: str, preferred: str, seats: int,
                   rooms: List[RoomRecord]) -> Optional[str]:
    """None when an active room of an allowed type seats ``seats``, else why not."""
    types = eligible_room_types(slot_type, preferred)
    usable = [r for r in rooms if r.is_active and (not types or r.room_type in types)]
    if any(r.capacity >= seats for r in usable):
        return None
    largest = max((r.capacity for r in usable), default=0)
    return (f'{slot_type}: needs {seats} seats in {"/".join(types) or "any room"}; '
            f'largest active one has {largest}')


# ── Entry point ──────────────────────────────────────────────────────────────

def run_precheck(term: Term,
                 batches: Iterable[BatchRecord],
                 students: Iterable[StudentRecord],
                 semesters: Iterable[Tuple[int, int]],
                 slots: Iterable[CourseSlotRecord],
                 courses: Iterable[CourseRecord],
                 offerings: Iterable[OfferingRecord],
                 rooms: Iterable[RoomRecord],
                 section_of: Callable,
                 weekly_hours: Optional[Dict[int, WeeklyHours]] = None,
                 capacity_buffer: float = CAPACITY_BUFFER) -> PrecheckReport:
    """Check that every core class of every section in ``term`` is schedulable.

    ``batches`` should be the running batches; ``semesters`` holds the
    (curriculum id, semester_no) pairs that exist; ``section_of`` is
    compute_section; ``weekly_hours`` maps course id to an explicit pattern.
    """
    semesters = set(semesters)
    if term.semester_type not in (ODD, EVEN):
        report = PrecheckReport(term=term, offering_year=None)
        report.issues.append(Issue(
            ERROR, UNSUPPORTED_TERM,
            f'{term.semester_type}: only Odd and Even semesters map to curriculum '
            f'semesters; summer terms are not supported.'))
        return report

    report = PrecheckReport(term=term,
                            offering_year=offering_year(term.academic_year, term.semester_type))
    issues = report.issues
    batches = list(batches)
    batch_by_id = {b.batch_id: b for b in batches}
    course_by_id = {c.course_id: c for c in courses}
    weekly_hours = weekly_hours or {}
    rooms = list(rooms)

    offerings_by_course: Dict[int, List[OfferingRecord]] = defaultdict(list)
    for off in offerings:
        if off.year == report.offering_year and off.semester_type == term.semester_type:
            offerings_by_course[off.course_id].append(off)

    slots_by_semester: Dict[Tuple[int, int], List[CourseSlotRecord]] = defaultdict(list)
    for slot in slots:
        slots_by_semester[(slot.curriculum_id, slot.semester_no)].append(slot)

    report.sections = build_sections(term, batches, students, section_of, issues)
    sections_by_batch: Dict[int, List[SectionGroup]] = defaultdict(list)
    for group in report.sections:
        sections_by_batch[group.batch_id].append(group)

    # course id -> [SectionGroup] needing it as a core class
    needed: Dict[int, List[SectionGroup]] = defaultdict(list)
    # course id -> ['<batch> (<slot type>)'] where its single-course slot was skipped
    skipped_single: Dict[int, List[str]] = defaultdict(list)
    for batch_id, groups in sections_by_batch.items():
        batch = batch_by_id[batch_id]
        sem_no = groups[0].semester_no
        if batch.curriculum_id is None:
            issues.append(Issue(ERROR, NO_CURRICULUM,
                                f'{batch.label}: running batch has no curriculum.',
                                batch=batch.label))
            continue
        if (batch.curriculum_id, sem_no) not in semesters:
            issues.append(Issue(
                ERROR, NO_SEMESTER,
                f'{batch.label}: curriculum has no semester {sem_no}.',
                batch=batch.label, detail={'semester_no': sem_no}))
            continue
        sem_slots = slots_by_semester.get((batch.curriculum_id, sem_no), [])
        if not sem_slots:
            issues.append(Issue(
                WARNING, EMPTY_SEMESTER,
                f'{batch.label}: semester {sem_no} has no course slots; nothing to schedule.',
                batch=batch.label, detail={'semester_no': sem_no}))
            continue
        for slot in sorted(sem_slots, key=lambda s: s.slot_id):
            if slot.slot_type in NON_CORE_SLOT_TYPES or len(slot.course_ids) > 1:
                why = (f'{slot.slot_type} slot' if slot.slot_type in NON_CORE_SLOT_TYPES
                       else f'choice of {len(slot.course_ids)} courses')
                issues.append(Issue(
                    WARNING, NON_CORE_SKIPPED,
                    f'{batch.label}: "{slot.name}" not scheduled ({why}); '
                    f'needs registrations or an elective basket.',
                    batch=batch.label,
                    detail={'slot_id': slot.slot_id, 'slot_type': slot.slot_type,
                            'courses': len(slot.course_ids)}))
                if len(slot.course_ids) == 1:
                    skipped_single[slot.course_ids[0]].append(
                        f'{batch.label} ({slot.slot_type})')
                continue
            if not slot.course_ids:
                issues.append(Issue(WARNING, EMPTY_CORE_SLOT,
                                    f'{batch.label}: core slot "{slot.name}" has no course.',
                                    batch=batch.label, detail={'slot_id': slot.slot_id}))
                continue
            # A course in several required slots of one batch is still one
            # class per section: add the batch's sections only once.
            if all(g.batch_id != batch_id for g in needed[slot.course_ids[0]]):
                needed[slot.course_ids[0]].extend(groups)

    # The curriculum does not say which slot type is right when the same course
    # is required in one batch and optional in another, so report it.
    for course_id in sorted(set(needed) & set(skipped_single),
                            key=lambda c: course_by_id[c].code):
        course = course_by_id[course_id]
        required_in = sorted({g.batch_label for g in needed[course_id]})
        issues.append(Issue(
            WARNING, SLOT_TYPE_INCONSISTENT,
            f'{course.code} {course.name}: required in {", ".join(required_in)} but in a '
            f'skipped slot for {", ".join(skipped_single[course_id])}; only the '
            f'required batches are checked.',
            course_code=course.code,
            detail={'required_in': required_in, 'skipped_in': skipped_single[course_id]}))

    instructor_hours: Counter = Counter()
    section_hours: Counter = Counter()
    used_offerings = set()
    for course_id in sorted(needed, key=lambda c: course_by_id[c].code):
        course = course_by_id[course_id]
        groups = needed[course_id]
        where = ', '.join(_group_name(g) for g in groups)

        hours = weekly_hours.get(course_id)
        source = 'requirement'
        if hours is None:
            hours, why = hours_from_course(course)
            source = 'course L/T/P'
            if hours is None:
                issues.append(Issue(
                    ERROR, HOURS_UNKNOWN,
                    f'{course.code} {course.name}: weekly hours unknown - {why}. '
                    f'Needed by {where}.',
                    course_code=course.code,
                    detail={'L': course.lecture_hours, 'T': course.tutorial_hours,
                            'P': course.practical_hours, 'credit': course.credit}))

        offers = offerings_by_course.get(course_id, [])
        chosen = {}
        # Sections are resolved per batch: the same label never spans batches,
        # but one offering list serves every batch taking the course.
        keys = [(g.batch_id, g.section) for g in groups]
        if len(set(g.section for g in groups)) != len(keys):
            reason = (f'{len(keys)} sections from several batches share section '
                      f'labels; offerings cannot tell them apart')
            if not offers:
                reason = 'no offering for this term'
        else:
            by_label, reason = resolve_section_offerings([g.section for g in groups], offers)
            chosen = {(g.batch_id, g.section): by_label[g.section]
                      for g in groups if g.section in by_label}
        if reason:
            code = MISSING_OFFERING if not offers else AMBIGUOUS_OFFERING
            issues.append(Issue(
                ERROR, code,
                f'{course.code} {course.name}: {reason}. Needed by {where}.',
                course_code=course.code,
                detail={'sections': [_group_name(g) for g in groups],
                        'offerings': [{'offering_id': o.offering_id,
                                       'instructor_id': o.instructor_id,
                                       'section_label': o.section_label} for o in offers]}))

        for group in groups:
            seats = seats_needed(group.size, capacity_buffer)
            if hours is not None and rooms:
                kinds = [(LECTURE, hours.lectures), (TUTORIAL, hours.tutorials),
                         (PRACTICAL, hours.practical_blocks)]
                for slot_type, count in kinds:
                    if not count:
                        continue
                    preferred = hours.preferred_room_type if slot_type == PRACTICAL else 'any'
                    short = room_shortfall(slot_type, preferred, seats, rooms)
                    if short:
                        issues.append(Issue(
                            ERROR, NO_ROOM_FITS,
                            f'{course.code} for {_group_name(group)} '
                            f'({group.size} students): {short}.',
                            batch=group.batch_label, section=group.section,
                            course_code=course.code,
                            detail={'students': group.size, 'seats_needed': seats,
                                    'slot_type': slot_type}))
            offering = chosen.get((group.batch_id, group.section))
            if hours is not None:
                section_hours[(group.batch_label, group.section)] += hours.total_hours
            if offering is None or hours is None:
                continue
            used_offerings.add(offering.offering_id)
            instructor_hours[offering.instructor_id] += hours.total_hours
            report.classes.append(PlannedClass(
                batch_id=group.batch_id, batch_label=group.batch_label,
                section=group.section, course_id=course_id, course_code=course.code,
                size=group.size, seats_needed=seats,
                offering_id=offering.offering_id, instructor_id=offering.instructor_id,
                hours=hours, hours_source=source))

    if not rooms:
        issues.append(Issue(ERROR, NO_ROOMS,
                            'No classrooms are defined; room capacity cannot be checked.'))

    for instructor, total in sorted(instructor_hours.items()):
        if total > WEEKLY_TEACHING_HOURS:
            issues.append(Issue(
                ERROR, INSTRUCTOR_OVERLOADED,
                f'Instructor {instructor}: {total} teaching hours a week exceed the '
                f'{WEEKLY_TEACHING_HOURS} available slots.',
                detail={'instructor_id': instructor, 'hours': total}))
    for (label, section), total in sorted(section_hours.items()):
        if total > WEEKLY_TEACHING_HOURS:
            issues.append(Issue(
                ERROR, SECTION_OVERLOADED,
                f'{label} section {section or "(whole batch)"}: {total} class hours a '
                f'week exceed the {WEEKLY_TEACHING_HOURS} available slots.',
                batch=label, section=section, detail={'hours': total}))

    needed_ids = set(needed)
    for course_id, offers in sorted(offerings_by_course.items()):
        if course_id not in needed_ids:
            continue
        for off in offers:
            if off.offering_id not in used_offerings and off.section_label:
                needed_labels = {g.section for g in needed[course_id]}
                if off.section_label not in needed_labels:
                    issues.append(Issue(
                        WARNING, UNUSED_OFFERING,
                        f'{course_by_id[course_id].code}: offering for section '
                        f'{off.section_label} ({off.instructor_id}) matches no section '
                        f'taking the course.',
                        course_code=course_by_id[course_id].code,
                        detail={'offering_id': off.offering_id}))
    return report


def _group_name(group: SectionGroup) -> str:
    return f'{group.batch_label} {group.section}' if group.section else group.batch_label
