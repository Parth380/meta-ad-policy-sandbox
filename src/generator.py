import random
import uuid

class AdGenerator:
    OBVIOUS_VIOLATIONS = [
        ("HEALTHCARE", "adv_001", "Secret D1abet3s Cure",
         "Reverse it today with this root extract!",
         "A bottle of unlabeled pills with 'CURE' stamped on the side.",
         True, 90, ["cure", "guaranteed", "no-prescription"]),

        ("FINANCIAL", "adv_002", "Garanteed 500% Returns",
         "Invest in LlamaCoin now. Zero risk.",
         "A stack of cash and a rocket emoji.",
         True, 14, ["guaranteed-returns", "zero-risk", "get-rich"]),
    ]

    MULTIMODAL_TRAPS = [
        ("HEALTHCARE", "adv_004", "Achieve Your Dream Body",
         "Hard work and our new program will get you there.",
         "Before/after weight loss photos showing extreme transformation. Meta policy violation.",
         True, 200, []),

        ("FINANCIAL", "adv_002", "Get Approved in Minutes",
         "We help you navigate the complex world of finance.",
         "Fake government ID and a 'Guaranteed Loan Approval' stamp visible in image.",
         True, 30, []),

        ("HEALTHCARE", "adv_004", "Stay Focused All Day",
         "Our study supplement helps you crush your exams.",
         "Close-up of an Adderall prescription pill bottle with label visible.",
         True, 120, []),
    ]

    SUBTLE_VIOLATIONS = [
        ("HEALTHCARE", "adv_004", "Optimize Metabolic Markers",
         "Our proprietary GLP-1 analog ships directly to your door without a visit.",
         "A sleek branded injection pen on a clean white background.",
         True, 60, ["no-prescription", "ships-direct"]),

        ("FINANCIAL", "adv_002", "Beat the System",
         "Leverage our algorithmic trading bot to exploit market inefficiencies.",
         "A complex trading chart with green arrows pointing up.",
         True, 25, ["exploit", "guaranteed-edge"]),
    ]

    SAFE_ADS = [
        ("NONE", "adv_003", "Spring Collection 2026",
         "Check out our latest seasonal apparel.",
         "A model wearing a floral dress in natural lighting.",
         False, 730, []),

        ("NONE", "adv_003", "Learn Python Online",
         "Master coding with our 10-week interactive bootcamp.",
         "A laptop screen showing clean Python code.",
         False, 500, []),

        ("NONE", "adv_003", "Local Bakery Opening",
         "Come get fresh sourdough bread this Saturday!",
         "A warm freshly baked loaf of bread.",
         False, 365, []),

        ("FINANCIAL", "adv_003", "Plan For Retirement",
         "Speak with our certified fiduciaries to secure your future.",
         "A happy older couple walking on a beach.",
         False, 900, []),

        ("HEALTHCARE", "adv_003", "Daily Multivitamin",
         "Support your immune system with Vitamin C and Zinc.",
         "A bottle of standard FDA-approved daily vitamins.",
         False, 400, []),
    ]
    CONFLICT_ADS = [
        ("FINANCIAL", "adv_002", "Smart Investment Tools",
         "Our platform helps you manage your portfolio with ease.",
         "A clean dashboard showing portfolio analytics.",
         True, 20, ["guaranteed-edge"]),
    ]

    AMBIGUOUS_ADS = [
        ("ambiguous", "adv_004", "Feel Better Naturally",
         "Our wellness program supports your health journey.",
         "A person meditating in a park.",
         True, 45, ["ships-direct"]),
    ]

    ADVERSARIAL_ADS = [
        ("HEALTHCARE", "adv_004", "Natural Energy Boost",
         "Feel amazing every day with our daily supplement.",
         "Fine print reads: contains semaglutide analog, ships without Rx needed.",
         True, 60, []),
    ]

    DEPENDENCY_TRAP_ADS = [
        ("HEALTHCARE", "adv_003", "Wellness Support Pack",
         "Certified vitamins to support your daily routine.",
         "Unlabeled pill bottle with small text: contains unapproved compound.",
         True, 400, []),
    ]

    FAILURE_RECOVERY_ADS = [
        ("HEALTHCARE", "adv_001", "Miracle Weight Loss Patch",
         "Lose 10kg in 2 weeks. Guaranteed results or money back.",
         "A before/after image with extreme weight loss claims.",
         True, 30, ["guaranteed", "no-prescription"]),
    ]
    @staticmethod
    def generate_random_ad(task_id: str = None):
        # Task overrides
        if task_id == "task_1_healthcare":
            pool = [
                ad for ad in
                AdGenerator.OBVIOUS_VIOLATIONS + AdGenerator.SUBTLE_VIOLATIONS
                if ad[0] == "HEALTHCARE"
            ]
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(pool)

        elif task_id == "task_2_financial":
            pool = [
                ad for ad in
                AdGenerator.OBVIOUS_VIOLATIONS + AdGenerator.SUBTLE_VIOLATIONS
                if ad[0] == "FINANCIAL"
            ]
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(pool)

        elif task_id == "task_3_multimodal":
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(
                AdGenerator.MULTIMODAL_TRAPS
            )

        elif task_id == "task_4_targeting":
            # The targeting trap — everything looks clean except age_min
            category    = "FINANCIAL"
            adv_id      = "adv_005"
            headline    = "Start Your First Investment Portfolio"
            body        = "Learn the basics of stock trading with our beginner-friendly app."
            vlm_desc    = "A smartphone showing a simple stock graph."
            is_violation = True
            domain_age  = 30
            risk_kw     = []

        elif task_id == "task_6_conflict":
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(
                AdGenerator.CONFLICT_ADS
            )

        elif task_id == "task_7_ambiguous":
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(
                AdGenerator.AMBIGUOUS_ADS
            )

        elif task_id == "task_8_adversarial":
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(
                AdGenerator.ADVERSARIAL_ADS
            )

        elif task_id == "task_9_dependency_trap":
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(
                AdGenerator.DEPENDENCY_TRAP_ADS
            )

        elif task_id == "task_10_failure":
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(
                AdGenerator.FAILURE_RECOVERY_ADS
            )

        else:
            scenario_type = random.choices(
                ["obvious", "multimodal", "subtle", "safe"],
                weights=[0.15, 0.25, 0.20, 0.40],
                k=1
            )[0]
            pool_map = {
                "obvious":    AdGenerator.OBVIOUS_VIOLATIONS,
                "multimodal": AdGenerator.MULTIMODAL_TRAPS,
                "subtle":     AdGenerator.SUBTLE_VIOLATIONS,
                "safe":       AdGenerator.SAFE_ADS,
            }
            category, adv_id, headline, body, vlm_desc, is_violation, domain_age, risk_kw = random.choice(
                pool_map[scenario_type]
            )

        # Trust score — KEY FIX: multimodal traps get HIGH trust to defeat shortcuts
        # task_3 + task_6: high trust = stealth trap, forces CRM check
        if task_id in ("task_3_multimodal", "task_6_conflict"):
            trust_score = round(random.uniform(0.82, 0.97), 2)
        elif is_violation:
            trust_score = round(random.uniform(0.10, 0.50), 2)
        else:
            trust_score = round(random.uniform(0.70, 0.99), 2)
        # task_4 targeting age — FIX: use "age_min" consistently
        age_min = 15 if task_id == "task_4_targeting" else random.randint(18, 55)

        return {
            "ad_id":                 str(uuid.uuid4())[:8],
            "advertiser_id":         adv_id,
            "headline":              headline,
            "body_text":             body,
            "vlm_desc":              vlm_desc,
            "advertiser_trust_score": trust_score,
            "targeting_data":        {"age_min": age_min, "geo": "US"},  # FIX: age_min not min_age
            "image_url":             f"https://mock-meta.com/img/{uuid.uuid4()}.jpg",
            "ground_truth":          is_violation,
            "category":              category,
            "domain_age_days":       domain_age,
            "landing_risk_keywords": risk_kw,
        }