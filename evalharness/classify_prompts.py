"""Prompt variants for the query-classifier A/B sweep (eval-owned).

``baseline`` is the *production* prompt, imported from ``rag.classify`` rather
than copied, so a change to production is automatically the thing being
compared against. Every other variant is derived from it and changes exactly
ONE thing, so the sweep attributes a difference to a single treatment:

  abstain         one extra rule: a wrong tag costs more than no tag
  examples        one appended block: 2 invented example questions per category
  rich-desc       the category description lines are replaced with
                  belongs / does-NOT-belong text + overlap disambiguation
  decision-rules  one appended block: ordered cues + a precedence order

``decision-rules`` deliberately stops at the precedence list and does NOT
restate the "return an empty list when nothing matches" fallback (baseline
already carries it) — otherwise it would silently also be an abstain arm and
the two treatments could not be told apart.

The example questions in ``examples`` are invented from insurance-domain
vocabulary. None is drawn from either reference set: reusing an evaluation
question as a prompt example would leak the answer key into the arm
(``tests/test_classify_eval.py`` enforces this with a substring check).

``VERIFY_SYSTEM_PROMPT`` / ``build_verify_user_message`` serve the second,
retraction-only call of the ``verify-2stage`` arm; they are not prompt variants
and never replace the classifier's own prompt.
"""
from __future__ import annotations

from rag.classify import CATEGORIES, _system_prompt

# --------------------------------------------------------------------------- #
# abstain
# --------------------------------------------------------------------------- #

ABSTAIN_RULE = (
    "- תג שגוי גרוע יותר מהיעדר תג: תג שגוי מפנה את החיפוש לתחום הלא נכון ומונע "
    "מציאת התשובה, בעוד רשימה ריקה מחפשת בכל התחומים. אם תחום הביטוח אינו נקבע "
    "באופן חד-משמעי מתוך נוסח השאלה, החזר רשימת categories ריקה."
)

# --------------------------------------------------------------------------- #
# examples — invented, never taken from a reference set
# --------------------------------------------------------------------------- #

EXAMPLE_QUESTIONS: dict[str, tuple[str, str]] = {
    "apartment": (
        "נזק לפרקט בסלון בעקבות דליפת מים מדירת השכן ממעל",
        "עד כמה מכוסים תכשיטים ששמורים בכספת בתוך הבית",
    ),
    "business": (
        "מלאי שנשרף במחסן של החנות שלי",
        "עובד שנפגע במשמרת לילה במפעל ואחריות המעביד",
    ),
    "car": (
        "השתתפות עצמית להחלפת שמשה קדמית ברכב פרטי",
        "גרירה מהכביש המהיר אחרי תקר בצמיג",
    ),
    "dental": (
        "השתלת שן טוחנת בשנה הראשונה למצטרף חדש",
        "כמה משלמים על טיפול שורש אצל רופא שבהסכם",
    ),
    "diseases-disabilities": (
        "פיצוי חד-פעמי עם אבחון של טרשת נפוצה",
        "שיעור התגמול כשנקבעה נכות צמיתה של ארבעים אחוז",
    ),
    "health": (
        "החזר על ניתוח אצל מנתח פרטי שאינו בהסכם",
        "תרופה שאינה כלולה בסל הבריאות הממלכתי",
    ),
    "life": (
        "החלפת מוטב אחרי שהפוליסה כבר נכנסה לתוקף",
        "מה קורה לפוליסה כשמפסיקים לשלם פרמיה חודשיים ברצף",
    ),
    "long-term-care": (
        "מתי מתחילים לשלם את הגמלה אחרי שנקבע מצב סיעודי",
        "שהייה במוסד סיעודי בלי שקדם לה אשפוז בבית חולים",
    ),
    "loss-of-working-ability": (
        "כמה זמן ממתינים עד לתשלום הראשון של הפיצוי החודשי",
        "עצמאי שאינו מסוגל להמשיך בעיסוק הספציפי שלו",
    ),
    "mortgage": (
        "האם ביטוח המבנה שדרש הבנק מכסה גם רעידת אדמה",
        "מה קורה לביטוח כשמסלקים את הלוואת הדיור מוקדם",
    ),
    "personal-accident": (
        "שבר ביד במשחק כדורגל חובבני בשכונה",
        "פיצוי יומי עבור ימי אשפוז בעקבות תאונה",
    ),
    "travel": (
        "ביטול טיסה בגלל מחלה של בן משפחה מדרגה ראשונה",
        "ציוד צילום שנגנב בשדה תעופה בחו\"ל",
    ),
}

# --------------------------------------------------------------------------- #
# rich-desc — belongs / does-NOT-belong, plus the three overlapping families
# --------------------------------------------------------------------------- #

RICH_DESCRIPTIONS: dict[str, str] = {
    "apartment": (
        "ביטוח דירה ותכולה. כולל: נזקי מים ורטיבות, שריפה, פריצה וגניבה מהדירה, "
        "רעידת אדמה, נזק שהדירה גרמה לצד שלישי, ביטוח המבנה והתכולה של דירת מגורים. "
        "לא כולל: ביטוח מבנה שנרכש כתנאי להלוואת דיור מהבנק (mortgage), עסק "
        "המתנהל מתוך הבית (business)."
    ),
    "business": (
        "ביטוח עסק. כולל: מבנה העסק, ציוד ומלאי, אובדן רווחים והפסקת פעילות, "
        "אחריות מקצועית, אחריות המוצר, חבות מעבידים כלפי עובדי העסק. "
        "לא כולל: פגיעה גופנית של המבוטח עצמו כאדם פרטי (personal-accident), "
        "אובדן הכנסה אישית של המבוטח (loss-of-working-ability)."
    ),
    "car": (
        "ביטוח רכב. כולל: חובה, מקיף וצד שלישי, נזקי התנגשות וגניבת רכב, שמשות, "
        "גרירה ושירותי דרך, ירידת ערך והשתתפות עצמית ברכב. "
        "לא כולל: פגיעה גופנית שמטופלת כתאונה אישית מחוץ להקשר הרכב "
        "(personal-accident), נזק לרכב שכור בחו\"ל בפוליסת נסיעות (travel)."
    ),
    "dental": (
        "ביטוח שיניים. כולל: טיפולים משמרים, טיפולי שורש, כתרים, גשרים, השתלות "
        "שיניים, יישור שיניים, רופאים בהסכם ומחוץ להסכם. "
        "לא כולל: ניתוחי פה ולסת שמבוצעים כניתוח רפואי (health), שבר בשיניים "
        "כתוצאה מתאונה כשהשאלה על פיצוי תאונתי (personal-accident)."
    ),
    "diseases-disabilities": (
        "ביטוח מחלות קשות ונכויות. כולל: תשלום פיצוי כספי, לרוב חד-פעמי, עם אבחון "
        "של מחלה קשה מתוך רשימה מוגדרת (סרטן, אירוע מוחי, התקף לב, טרשת נפוצה "
        "וכדומה) או עם קביעת אחוזי נכות. הפיצוי אינו תלוי בהוצאה רפואית בפועל. "
        "לא כולל: מימון או החזר של הטיפול הרפואי עצמו (health), גמלה חודשית "
        "לתפקוד יומיומי (long-term-care)."
    ),
    "health": (
        "ביטוח בריאות. כולל: מימון והחזר של הוצאות רפואיות — ניתוחים בארץ ובחו\"ל, "
        "מנתחים בהסכם ושלא בהסכם, תרופות מחוץ לסל, השתלות, אבחונים, התייעצויות "
        "עם רופאים מומחים, כתבי שירות. "
        "לא כולל: פיצוי חד-פעמי בעקבות אבחון מחלה קשה (diseases-disabilities), "
        "גמלה חודשית לתלוי בעזרת הזולת (long-term-care), החלפת שכר בזמן אי-כושר "
        "(loss-of-working-ability)."
    ),
    "life": (
        "ביטוח חיים. כולל: תגמול למוטבים במקרה מוות, ריסק זמני, ערכי פדיון וסילוק, "
        "שינוי מוטבים, חיתום רפואי והצהרת בריאות בעת ההצטרפות, תשלום פרמיה והפסקתה. "
        "לא כולל: ביטוח חיים שהבנק דרש כבטוחה להלוואת דיור (mortgage), מוות "
        "שנגרם מתאונה כשהשאלה על פיצוי תאונתי (personal-accident)."
    ),
    "long-term-care": (
        "ביטוח סיעודי. כולל: גמלה חודשית למי שאינו מסוגל לבצע פעולות יומיומיות "
        "בסיסיות (לקום, להתלבש, להתרחץ, לאכול, שליטה על סוגרים, ניידות) או שנמצא "
        "במצב של תשישות נפש, שהייה במוסד סיעודי, טיפול סיעודי בבית, תקופת המתנה "
        "לגמלה ותקופת התשלום. "
        "לא כולל: מימון הטיפול הרפואי עצמו (health), פיצוי חד-פעמי על נכות "
        "(diseases-disabilities)."
    ),
    "loss-of-working-ability": (
        "ביטוח אובדן כושר עבודה. כולל: תגמול חודשי שמחליף הכנסה כשהמבוטח אינו כשיר "
        "לעבוד — מכל סיבה שהיא, מחלה או תאונה — הגדרת עיסוק ספציפי מול עיסוק סביר, "
        "תקופת המתנה, שחרור מתשלום פרמיה, אובדן כושר חלקי או מלא. "
        "לא כולל: פיצוי חד-פעמי על נכות צמיתה (diseases-disabilities), פיצוי "
        "בעקבות אירוע תאונתי נקודתי (personal-accident)."
    ),
    "mortgage": (
        "ביטוח משכנתא. כולל: ביטוח חיים וביטוח מבנה שהבנק דורש כתנאי להלוואת דיור, "
        "כשהמוטב הוא הבנק עד גובה יתרת ההלוואה, סילוק מוקדם של ההלוואה, מחזור "
        "משכנתא, נכס משועבד, שינוי גובה הכיסוי לאורך חיי ההלוואה. "
        "לא כולל: ביטוח חיים עצמאי שהמוטב בו נבחר על ידי הלקוח (life), ביטוח דירה "
        "ותכולה שנרכש מרצון (apartment)."
    ),
    "personal-accident": (
        "ביטוח תאונות אישיות. כולל: פיצוי בעקבות אירוע תאונתי חד-פעמי ופתאומי — "
        "שבר, כוויה, נכות או מוות מתאונה — פיצוי יומי עבור ימי אשפוז או ימי אי-כושר "
        "לאחר תאונה, תאונות ספורט ופנאי. "
        "לא כולל: מחלה שאינה תוצאה של תאונה (health או diseases-disabilities), "
        "תגמול חודשי ארוך-טווח שמחליף שכר (loss-of-working-ability)."
    ),
    "travel": (
        "ביטוח נסיעות לחו\"ל. כולל: הוצאות רפואיות בחו\"ל, פינוי והטסה רפואית, "
        "ביטול או קיצור נסיעה, כבודה ומטען, מכשירים אלקטרוניים בנסיעה, ספורט "
        "אתגרי בחו\"ל, הארכת תקופת הביטוח בזמן שהייה בחו\"ל. "
        "לא כולל: טיפול רפואי בישראל (health), נזק לתכולת הבית בזמן שהמבוטח "
        "בחו\"ל (apartment)."
    ),
}

DISAMBIGUATION = """הבחנות בין משפחות תחומים חופפות:
- health מול diseases-disabilities מול long-term-care: health משלם עבור טיפול רפואי (מה ההוצאה ומי מממן אותה). diseases-disabilities משלם פיצוי עם אבחון או עם קביעת נכות, בלי קשר להוצאה בפועל. long-term-care משלם גמלה חודשית למי שאיבד עצמאות תפקודית יומיומית.
- life מול mortgage: אם מוזכרים משכנתא, בנק, הלוואת דיור, יתרת הלוואה או נכס משועבד — mortgage. ביטוח חיים שהמוטב בו נבחר על ידי המבוטח ולא משועבד לבנק — life.
- personal-accident מול loss-of-working-ability: personal-accident מתייחס לאירוע תאונתי נקודתי ולפיצוי שנגזר ממנו. loss-of-working-ability מתייחס ליכולת להמשיך להתפרנס לאורך זמן, ללא תלות בשאלה אם הסיבה היא תאונה או מחלה."""

# --------------------------------------------------------------------------- #
# decision-rules
# --------------------------------------------------------------------------- #

DECISION_RULES = """סדר ההכרעה בתיוג — עבור על הכללים לפי סדרם ועצור בכלל הראשון שמכריע:
1. שם מוצר או שם פוליסה שמופיע במפורש בשאלה (למשל "ביטוח נסיעות", "פוליסת הרכב", "ביטוח השיניים", "הפוליסה הסיעודית") קובע את התחום, גם אם תוכן השאלה נשמע כללי.
2. אין שם מוצר — הכרע לפי מצב המבוטח שהשאלה מתארת: מה קרה לו, מה הוא מבקש ומי משלם. בקשת החזר על הוצאה רפואית ← health. פיצוי בעקבות אבחון מחלה או קביעת נכות ← diseases-disabilities. תגמול חודשי שמחליף שכר עבודה ← loss-of-working-ability. גמלה למי שתלוי בעזרת הזולת בפעולות יומיום ← long-term-care. נזק לרכוש בבית ← apartment.
3. שני תחומים עדיין מתאימים — הכרע לפי סדר הקדימויות הבא:
   - הוזכרו משכנתא, בנק או הלוואת דיור ← mortgage גובר על life ועל apartment.
   - האירוע קרה בחו"ל במהלך נסיעה ← travel גובר על health ועל apartment.
   - הזכאות נובעת מאירוע תאונתי נקודתי ← personal-accident גובר על health.
   - מדובר בתלות בעזרת הזולת בפעולות יומיום ← long-term-care גובר על health.
   - מדובר בפיצוי חד-פעמי עם אבחון מחלה קשה מוגדרת ← diseases-disabilities גובר על health."""


# --------------------------------------------------------------------------- #
# Variant builders
# --------------------------------------------------------------------------- #


def _category_lines() -> str:
    """The exact category block the production prompt embeds."""
    return "\n".join(f"- {cid}: {desc}" for cid, desc in CATEGORIES.items())


def _baseline() -> str:
    return _system_prompt()


def _abstain() -> str:
    return _baseline() + "\n" + ABSTAIN_RULE


def _examples() -> str:
    lines = ["דוגמאות לשיוך נכון (נוסח מקוצר של פניות אופייניות):"]
    for cid, examples in EXAMPLE_QUESTIONS.items():
        for example in examples:
            lines.append(f'- "{example}" ← {cid}')
    return _baseline() + "\n\n" + "\n".join(lines)


def _rich_desc() -> str:
    base = _baseline()
    plain = _category_lines()
    if base.count(plain) != 1:
        raise RuntimeError(
            "production prompt no longer embeds the plain category block verbatim — "
            "rich-desc cannot substitute it; update evalharness/classify_prompts.py"
        )
    rich = "\n".join(f"- {cid}: {RICH_DESCRIPTIONS[cid]}" for cid in CATEGORIES)
    return base.replace(plain, rich + "\n\n" + DISAMBIGUATION)


def _decision_rules() -> str:
    return _baseline() + "\n\n" + DECISION_RULES


PROMPT_VARIANTS = {
    "baseline": _baseline,
    "abstain": _abstain,
    "examples": _examples,
    "rich-desc": _rich_desc,
    "decision-rules": _decision_rules,
}


def build_prompt(name: str) -> str:
    """Render one prompt variant. ``baseline`` is byte-identical to production."""
    if name not in PROMPT_VARIANTS:
        raise KeyError(f"unknown prompt variant {name!r}; have: {sorted(PROMPT_VARIANTS)}")
    return PROMPT_VARIANTS[name]()


# --------------------------------------------------------------------------- #
# verify-2stage second call (retraction only — it may drop tags, never add)
# --------------------------------------------------------------------------- #

VERIFY_SYSTEM_PROMPT = """אתה בודק תיוג של פניית לקוח בתחום הביטוח. קיבלת שאלה, רשימת תגי תחום שהוצעו עבורה, וראיות מהאינדקס: התפלגות התחומים של תוצאות החיפוש המובילות וקטעים מהן.

החלט עבור כל תג שהוצע האם להשאיר אותו. אינך רשאי להוסיף תגים חדשים.
- השאר תג רק אם השאלה עצמה שייכת בבירור לתחום.
- תג שגוי מפנה את החיפוש לתחום הלא נכון ומונע מציאת התשובה, ולכן עדיף להסיר תג מוטל בספק מאשר להשאיר אותו.
- הראיות הן אינדיקציה בלבד ואינן מכריעות: מנוע החיפוש עלול להחזיר קטעים מתחום שכן רק בגלל ניסוח דומה.

החזר JSON בלבד, ללא טקסט נוסף: {"keep": ["<id>", ...]}"""


def build_verify_user_message(question: str, categories: list[str], hint: str) -> str:
    tags = ", ".join(categories) if categories else "(אין)"
    return (
        f"שאלה: {question}\n\n"
        f"תגים שהוצעו: {tags}\n\n"
        "--- ראיות מהאינדקס ---\n"
        f"{hint}\n"
        "--- סוף ראיות ---"
    )
