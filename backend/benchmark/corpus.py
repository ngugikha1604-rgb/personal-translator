"""corpus.py — 100 benchmark conversations for Copilot quality measurement.

Each conversation: list of {"speaker": "other"|"user", "text": "..."}
Only "other" turns are analyzed by the copilot.
"user" turns are filler for context.

Fields: id, category, label, turns
"""

benchmark_corpus = [
    # ── Category A: Networking (25 conversations) ──────────────────────────
    *[
        {"id": f"A{i}", "category": "networking", "label": label, "turns": turns}
        for i, (label, turns) in enumerate([
            # A1-A5: Conference meet-and-greet
            ("opening with small talk", [
                {"speaker": "other", "text": "Hey, nice to meet you! So what brings you here?"},
                {"speaker": "user", "text": "Just here to see what's going on."},
                {"speaker": "other", "text": "That's cool. What do you work on?"},
                {"speaker": "user", "text": "AI stuff, building language models."},
                {"speaker": "other", "text": "Oh nice! What got you into that?"},
                {"speaker": "user", "text": "Found them fascinating, changed how I code."},
            ]),
            ("skill probe after introduction", [
                {"speaker": "other", "text": "I heard you work on AI. What kind of stuff do you build?"},
                {"speaker": "user", "text": "Mostly conversational AI and copilots."},
                {"speaker": "other", "text": "Are you using RAG or fine-tuning?"},
                {"speaker": "user", "text": "Mostly RAG, some fine-tuning for specific tasks."},
                {"speaker": "other", "text": "How do you handle hallucinations?"},
            ]),
            ("indirect recruiting", [
                {"speaker": "other", "text": "Your talk on real-time systems was really impressive."},
                {"speaker": "user", "text": "Thanks, it was a fun project."},
                {"speaker": "other", "text": "We're actually looking for people with that kind of background."},
                {"speaker": "user", "text": "Oh interesting, what kind of role?"},
                {"speaker": "other", "text": "Senior engineer on our ML inference team. Does that sound like something you'd consider?"},
            ]),
            ("conference logistics", [
                {"speaker": "other", "text": "Are you enjoying the conference so far?"},
                {"speaker": "user", "text": "Yeah, some great talks."},
                {"speaker": "other", "text": "Which session was your favorite?"},
                {"speaker": "user", "text": "The keynote on multimodal models was eye-opening."},
                {"speaker": "other", "text": "Oh I missed that one, was it packed?"},
            ]),
            ("finding collaborators", [
                {"speaker": "other", "text": "I'm looking for someone to collaborate on an open source project."},
                {"speaker": "user", "text": "What kind of project?"},
                {"speaker": "other", "text": "A local-first AI assistant that works offline. Are you interested?"},
                {"speaker": "user", "text": "That sounds really interesting, what's your timeline?"},
                {"speaker": "other", "text": "Hoping to have a prototype in about three months."},
            ]),
            # A6-A10: Industry event
            ("genuine curiosity about work", [
                {"speaker": "other", "text": "So what does your company actually do?"},
                {"speaker": "user", "text": "We build AI tools for remote teams."},
                {"speaker": "other", "text": "How do you guys make money from that?"},
                {"speaker": "user", "text": "Enterprise subscription mostly."},
                {"speaker": "other", "text": "Interesting, must be a competitive space."},
            ]),
            ("testing skepticism", [
                {"speaker": "other", "text": "You mentioned you use GPT-4. How do you handle the cost?"},
                {"speaker": "user", "text": "We batch process and cache aggressively."},
                {"speaker": "other", "text": "And what about latency? Can you actually get sub-second responses?"},
                {"speaker": "user", "text": "We get about 800ms on average for most queries."},
                {"speaker": "other", "text": "At what cost per request though? I've seen those bills."},
            ]),
            ("polite interest fading", [
                {"speaker": "other", "text": "That's really fascinating work."},
                {"speaker": "user", "text": "Thanks! I could talk about it for hours."},
                {"speaker": "other", "text": "Uh-huh. Anyway, have you been to this event before?"},
                {"speaker": "user", "text": "No, first time. You?"},
                {"speaker": "other", "text": "Yeah, couple times. It's usually pretty good."},
            ]),
            ("follow-up from earlier conversation", [
                {"speaker": "other", "text": "Hey, we were talking earlier about the recommendation system. I had another thought."},
                {"speaker": "user", "text": "Oh sure, what's on your mind?"},
                {"speaker": "other", "text": "What if instead of collaborative filtering, you just used a graph-based approach?"},
                {"speaker": "user", "text": "We actually tried that, but the cold start problem was worse."},
                {"speaker": "other", "text": "Did you try adding content-based features to the graph, though?"},
            ]),
            ("direct call-out", [
                {"speaker": "other", "text": "Wait, I remember you from the last conference. You had that demo with the voice bot, right?"},
                {"speaker": "user", "text": "Yeah, that was me! Can't believe you remember."},
                {"speaker": "other", "text": "It was hard to forget. Did you ever get that VC funding you were looking for?"},
                {"speaker": "user", "text": "Actually yes, we closed the round last quarter."},
            ]),
            # A11-A15: Casual coworker
            ("casual catchup with colleague", [
                {"speaker": "other", "text": "Hey stranger, haven't seen you in a while. How've you been?"},
                {"speaker": "user", "text": "Pretty good, been swamped with the new project."},
                {"speaker": "other", "text": "Oh yeah? What project is that?"},
                {"speaker": "user", "text": "Rewriting our entire data pipeline."},
                {"speaker": "other", "text": "Oof, that sounds like a lot of work. How's it going?"},
            ]),
            ("lunch invitation", [
                {"speaker": "other", "text": "I'm heading to get lunch, want to join?"},
                {"speaker": "user", "text": "Sure, I could eat."},
                {"speaker": "other", "text": "Any preferences? There's that Thai place or the usual sandwich spot."},
                {"speaker": "user", "text": "Thai sounds good."},
                {"speaker": "other", "text": "Cool, let's go before it gets crowded."},
            ]),
            ("venting about work", [
                {"speaker": "other", "text": "Man, this week has been brutal. The deadline keeps moving up."},
                {"speaker": "user", "text": "I know the feeling, our team has the same problem."},
                {"speaker": "other", "text": "How do you guys deal with the pressure without burning out?"},
                {"speaker": "user", "text": "We try to be realistic about what we can ship."},
                {"speaker": "other", "text": "Our manager doesn't seem to get the concept of realistic."},
            ]),
            ("back from vacation", [
                {"speaker": "other", "text": "Welcome back! How was your trip?"},
                {"speaker": "user", "text": "Amazing, thanks! Japan was incredible."},
                {"speaker": "other", "text": "Oh nice! Where all did you go?"},
                {"speaker": "user", "text": "Tokyo, Kyoto, and a few days in Osaka."},
                {"speaker": "other", "text": "Did you make it to any of the tech meetups there?"},
            ]),
            ("hobby crossover", [
                {"speaker": "other", "text": "I heard you play guitar. What kind of music do you play?"},
                {"speaker": "user", "text": "Mostly blues and classic rock."},
                {"speaker": "other", "text": "Nice! Do you ever jam with other people?"},
                {"speaker": "user", "text": "Sometimes, but mostly just for myself."},
                {"speaker": "other", "text": "We should jam sometime, I play drums."},
            ]),
            # A16-A20: Cold introduction
            ("introduction through mutual friend", [
                {"speaker": "other", "text": "John said I should talk to you. He mentioned you work on something similar to what I do."},
                {"speaker": "user", "text": "Oh nice, what do you work on?"},
                {"speaker": "other", "text": "I run an ML platform team at a mid-size fintech company."},
                {"speaker": "user", "text": "That sounds great, we're actually looking for partners in that space."},
                {"speaker": "other", "text": "What kind of integration did you have in mind?"},
            ]),
            ("standup interaction", [
                {"speaker": "other", "text": "What did you work on yesterday?"},
                {"speaker": "user", "text": "Finished the API refactor and started writing tests."},
                {"speaker": "other", "text": "Any blockers I should know about?"},
                {"speaker": "user", "text": "The CI pipeline is still flaky, slows things down."},
                {"speaker": "other", "text": "I can help look at that after standup if you want."},
            ]),
            ("introducing a candidate", [
                {"speaker": "other", "text": "Hey, I want to introduce you to Sarah. She's been looking at similar problems in computer vision."},
                {"speaker": "user", "text": "Great, always good to meet others in the space."},
                {"speaker": "other", "text": "Sarah, this is Alex — the one I was telling you about who built that real-time detection system."},
                {"speaker": "other", "text": "So Alex, what kind of models are you using these days?"},
            ]),
            ("technically probing", [
                {"speaker": "other", "text": "You mentioned you use distributed training. What framework?"},
                {"speaker": "user", "text": "Mostly PyTorch DDP, some FSDP for large models."},
                {"speaker": "other", "text": "Have you tried DeepSpeed? We found it way more efficient."},
                {"speaker": "user", "text": "We benchmarked it but the integration cost wasn't worth it for our scale."},
                {"speaker": "other", "text": "Fair enough. What size models are we talking about?"},
            ]),
            ("closing a conversation gracefully", [
                {"speaker": "other", "text": "Well, I should probably mingle a bit more, but it was great talking with you."},
                {"speaker": "user", "text": "You too, really enjoyed the conversation."},
                {"speaker": "other", "text": "Let me grab your card, I'll follow up about that collaboration idea."},
                {"speaker": "user", "text": "Sounds good, looking forward to it."},
                {"speaker": "other", "text": "Talk soon!"},
            ]),
            # A21-A25: Mixed
            ("compliment then redirect", [
                {"speaker": "other", "text": "Your presentation was really well structured."},
                {"speaker": "user", "text": "Thanks, I spent way too long on it."},
                {"speaker": "other", "text": "But I'm curious about one thing — you said real-time was important, but your demo had a noticeable delay. Was that intentional?"},
            ]),
            ("trying to find common ground", [
                {"speaker": "other", "text": "I'm coming at this from the biology side, but the overlap is fascinating."},
                {"speaker": "user", "text": "Totally, there's a lot of cross-pollination happening."},
                {"speaker": "other", "text": "Have you ever thought about applying your techniques to genomic data?"},
            ]),
            ("after-hours social", [
                {"speaker": "other", "text": "A bunch of us are heading to a bar nearby, wanna come?"},
                {"speaker": "user", "text": "Sure, why not, I'm done with talks for today."},
                {"speaker": "other", "text": "Great! Do you know the others, or should I introduce you on the way?"},
            ]),
            ("debrief after a panel", [
                {"speaker": "other", "text": "What did you think of the panel discussion?"},
                {"speaker": "user", "text": "Interesting but a bit surface-level. Wanted more depth on the technical challenges."},
                {"speaker": "other", "text": "Yeah, they didn't really address the scaling issues, which is the hard part."},
            ]),
            ("genuine offer to help", [
                {"speaker": "other", "text": "I noticed you were asking about quantization strategies. I actually wrote a paper on that a couple years ago."},
                {"speaker": "user", "text": "Oh that's perfect, would you be open to sharing it?"},
                {"speaker": "other", "text": "Absolutely, let me send you the link and I'm happy to chat through it more if you have questions."},
            ]),
        ])
    ],

    # ── Category B: Interview (25 conversations) ───────────────────────────
    *[
        {"id": f"B{i}", "category": "interview", "label": label, "turns": turns}
        for i, (label, turns) in enumerate([
            ("general background probe", [
                {"speaker": "other", "text": "So tell me a bit about yourself and your background."},
                {"speaker": "user", "text": "I've been in software engineering for about 6 years, mostly in ML."},
                {"speaker": "other", "text": "What drew you to machine learning specifically?"},
                {"speaker": "user", "text": "I was always fascinated by how recommendation systems worked."},
                {"speaker": "other", "text": "That's interesting. What's the most complex system you've designed?"},
            ]),
            ("depth check on claimed experience", [
                {"speaker": "other", "text": "Your resume says you're proficient in PyTorch. Can you walk me through how you'd implement a custom training loop?"},
                {"speaker": "user", "text": "Sure, I'd start with the DataLoader, model definition, then the training loop with loss calculation and backprop."},
                {"speaker": "other", "text": "How do you handle gradient accumulation across multiple GPUs?"},
                {"speaker": "user", "text": "Use the no_sync context manager and manually accumulate steps."},
                {"speaker": "other", "text": "And what about mixed precision? Have you worked with AMP?"},
            ]),
            ("behavioral: conflict resolution", [
                {"speaker": "other", "text": "Tell me about a time you disagreed with a technical decision."},
                {"speaker": "user", "text": "I once argued against migrating to microservices because the team wasn't ready."},
                {"speaker": "other", "text": "What happened? Did they go with microservices anyway?"},
                {"speaker": "user", "text": "We compromised — extracted the most critical module first as a pilot."},
                {"speaker": "other", "text": "Looking back, do you think you were right, or was the migration the right call?"},
            ]),
            ("experience calibration", [
                {"speaker": "other", "text": "How long have you been working with Kubernetes?"},
                {"speaker": "user", "text": "About 3 years, in production for 2."},
                {"speaker": "other", "text": "Have you managed clusters at scale, say more than 500 nodes?"},
                {"speaker": "user", "text": "Not that many, the largest was about 150."},
                {"speaker": "other", "text": "How did you handle networking and service discovery at that scale?"},
            ]),
            ("motivation check", [
                {"speaker": "other", "text": "Why do you want to leave your current position?"},
                {"speaker": "user", "text": "I'm looking for more challenging problems and a stronger research culture."},
                {"speaker": "other", "text": "What specifically about our research culture appeals to you?"},
                {"speaker": "user", "text": "The papers your team has published on efficient inference are exactly the kind of work I want to do."},
            ]),
            ("weakness question", [
                {"speaker": "other", "text": "What would you say is your biggest weakness as an engineer?"},
                {"speaker": "user", "text": "I tend to over-engineer solutions and then have to pull back."},
                {"speaker": "other", "text": "Can you give me a concrete example of where that happened?"},
            ]),
            ("future goals", [
                {"speaker": "other", "text": "Where do you see yourself in five years?"},
                {"speaker": "user", "text": "I'd like to lead a team working on applied ML research."},
                {"speaker": "other", "text": "What kind of research interests you most?"},
                {"speaker": "user", "text": "Model efficiency and deployment at the edge."},
            ]),
            ("system design question", [
                {"speaker": "other", "text": "Let's say you need to design a real-time recommendation system for a video platform. Walk me through your approach."},
                {"speaker": "user", "text": "I'd separate the data pipeline from the inference path."},
                {"speaker": "other", "text": "How would you handle the cold start problem for new users?"},
                {"speaker": "user", "text": "Use content-based features as a fallback until there's enough interaction history."},
            ]),
            ("technical disagreement scenario", [
                {"speaker": "other", "text": "Your teammate wants to use PostgreSQL for a time-series workload. You disagree. How do you handle it?"},
                {"speaker": "user", "text": "I'd show benchmark data comparing it with specialized time-series databases."},
                {"speaker": "other", "text": "What if they still insist after seeing the data?"},
            ]),
            ("salary expectations", [
                {"speaker": "other", "text": "What salary range are you expecting for this role?"},
                {"speaker": "user", "text": "Based on the market and my experience, around 180 to 220."},
                {"speaker": "other", "text": "And how flexible are you on equity vs cash?"},
            ]),
            ("project deep dive", [
                {"speaker": "other", "text": "Walk me through the most technically challenging project you've led."},
                {"speaker": "user", "text": "I designed a real-time fraud detection system processing 10K events per second."},
                {"speaker": "other", "text": "What was the hardest technical problem you had to solve?"},
                {"speaker": "user", "text": "Keeping latency under 50ms while maintaining high recall."},
                {"speaker": "other", "text": "How did you eventually solve it?"},
            ]),
            ("self-assessment", [
                {"speaker": "other", "text": "On a scale of 1 to 10, how would you rate your Python skills?"},
                {"speaker": "user", "text": "Probably a solid 8, been using it daily for years."},
                {"speaker": "other", "text": "And where would you say you still have gaps?"},
            ]),
            ("ambiguity handling", [
                {"speaker": "other", "text": "You're given a vague requirement: 'make the system faster'. How do you approach it?"},
                {"speaker": "user", "text": "First I'd measure current performance to identify bottlenecks."},
                {"speaker": "other", "text": "What if there's no monitoring in place and you need to start from scratch?"},
            ]),
            ("prioritization under pressure", [
                {"speaker": "other", "text": "You have three urgent bugs and limited time. How do you decide what to fix first?"},
                {"speaker": "user", "text": "I assess impact on users and business continuity."},
                {"speaker": "other", "text": "What if one bug affects 5% of users but has executive visibility, and another affects 40% but isn't visible?"},
            ]),
            ("giving feedback", [
                {"speaker": "other", "text": "How do you give constructive feedback to a junior engineer who's struggling?"},
                {"speaker": "user", "text": "I focus on specific behaviors, not personality, and frame it as coaching."},
                {"speaker": "other", "text": "Can you give me an actual example of feedback you've given recently?"},
            ]),
            ("rejection handling", [
                {"speaker": "other", "text": "How do you react when your design proposal gets rejected by the team?"},
                {"speaker": "user", "text": "I try to understand the concerns and iterate on the proposal."},
                {"speaker": "other", "text": "Have you ever been in a situation where your solution was clearly better but the team chose wrong?"},
            ]),
            ("learning ability", [
                {"speaker": "other", "text": "You need to pick up a technology you've never used before for a project starting next week. What's your approach?"},
                {"speaker": "user", "text": "I'd find the fastest path to a working prototype and iterate."},
                {"speaker": "other", "text": "What specific resources do you typically rely on?"},
            ]),
            ("management style", [
                {"speaker": "other", "text": "How do you manage underperforming team members?"},
                {"speaker": "user", "text": "I start by understanding whether it's a motivation issue or a skill issue."},
                {"speaker": "other", "text": "And if it's a skill issue, what's your approach?"},
            ]),
            ("work-life boundary", [
                {"speaker": "other", "text": "Our team occasionally works weekends during crunch time. How do you feel about that?"},
                {"speaker": "user", "text": "I'm okay with occasional crunch as long as it's the exception, not the norm."},
                {"speaker": "other", "text": "How do you maintain balance when things get intense?"},
            ]),
            ("closing the interview", [
                {"speaker": "other", "text": "Do you have any questions for me about the role or the team?"},
                {"speaker": "user", "text": "Yes, what does success look like in the first 90 days?"},
            ]),
            ("mixed signals", [
                {"speaker": "other", "text": "Your background is really impressive. But I'm concerned about your lack of experience in our specific domain."},
                {"speaker": "user", "text": "I've jumped domains before and picked things up quickly."},
                {"speaker": "other", "text": "That's fair. How would you go about ramping up here?"},
            ]),
            ("cultural fit probe", [
                {"speaker": "other", "text": "What kind of work environment helps you do your best work?"},
                {"speaker": "user", "text": "I thrive when there's autonomy and clear goals, but also good collaboration."},
                {"speaker": "other", "text": "How do you handle a team where people have very different communication styles?"},
            ]),
            ("probing for leadership potential", [
                {"speaker": "other", "text": "Have you mentored junior engineers before?"},
                {"speaker": "user", "text": "Yes, I've mentored three engineers over the past two years."},
                {"speaker": "other", "text": "What was your approach to helping them grow?"},
            ]),
            ("reference check simulation", [
                {"speaker": "other", "text": "What would your previous manager say is your biggest area for growth?"},
                {"speaker": "user", "text": "Probably that I take on too much and don't delegate enough."},
                {"speaker": "other", "text": "Is that something you're actively working on?"},
            ]),
        ])
    ],

    # ── Category C: Technical discussion (25 conversations) ────────────────
    *[
        {"id": f"C{i}", "category": "technical", "label": label, "turns": turns}
        for i, (label, turns) in enumerate([
            ("architecture: monolith vs microservices", [
                {"speaker": "other", "text": "We're debating whether to break our monolith into microservices. What's your take?"},
                {"speaker": "user", "text": "It depends on your team size and deployment frequency."},
                {"speaker": "other", "text": "We have about 20 engineers and deploy twice a week."},
                {"speaker": "user", "text": "At that size, you're probably fine with a well-structured monolith."},
                {"speaker": "other", "text": "Even if we're planning to scale the team to 50?"},
            ]),
            ("bug investigation: memory leak", [
                {"speaker": "other", "text": "Our service keeps running out of memory after about 6 hours."},
                {"speaker": "user", "text": "Have you looked at heap dumps to see what's growing?"},
                {"speaker": "other", "text": "We took a dump but it's 8GB, hard to analyze."},
                {"speaker": "user", "text": "You could use a smaller interval between dumps to narrow down the pattern."},
                {"speaker": "other", "text": "Good idea. What tools do you recommend for analysis?"},
            ]),
            ("tool: React vs Vue", [
                {"speaker": "other", "text": "We're starting a new frontend project and can't decide between React and Vue."},
                {"speaker": "user", "text": "What's your team's experience with either?"},
                {"speaker": "other", "text": "Mostly React, but a couple people prefer Vue."},
                {"speaker": "user", "text": "Stick with React then, the hiring market is better too."},
                {"speaker": "other", "text": "But Vue is supposed to be easier to learn for new devs, isn't it?"},
            ]),
            ("CI/CD pipeline optimization", [
                {"speaker": "other", "text": "Our CI pipeline takes 45 minutes. We need to bring it down."},
                {"speaker": "user", "text": "Have you profiled what's taking the longest?"},
                {"speaker": "other", "text": "Integration tests take 25 minutes, and linting takes 10."},
                {"speaker": "user", "text": "You could parallelize tests and cache more aggressively."},
                {"speaker": "other", "text": "We tried parallelizing but some tests share state."},
            ]),
            ("production incident: traffic spike", [
                {"speaker": "other", "text": "We're getting a traffic spike 3x our normal load and the database is melting."},
                {"speaker": "user", "text": "Can you temporarily scale up read replicas?"},
                {"speaker": "other", "text": "We have auto-scaling but it's not keeping up."},
                {"speaker": "user", "text": "Might need to look at query optimization or add caching."},
                {"speaker": "other", "text": "The queries are already fairly optimized. I think it's connection pool exhaustion."},
            ]),
            ("database: SQL vs NoSQL", [
                {"speaker": "other", "text": "We're choosing a database for a new analytics platform. SQL or NoSQL?"},
                {"speaker": "user", "text": "What kind of queries do you need to support?"},
                {"speaker": "other", "text": "Mostly time-series aggregations and ad-hoc queries."},
                {"speaker": "user", "text": "A time-series optimized database like ClickHouse or TimescaleDB might be best."},
                {"speaker": "other", "text": "But we also need transactional consistency for some parts."},
            ]),
            ("deployment strategy: blue-green vs canary", [
                {"speaker": "other", "text": "We're setting up a new deployment pipeline. Blue-green or canary?"},
                {"speaker": "user", "text": "Blue-green is simpler but more expensive on infrastructure."},
                {"speaker": "other", "text": "Cost isn't our main concern, safety is."},
                {"speaker": "user", "text": "Then blue-green with automated rollback on error rates."},
                {"speaker": "other", "text": "How do you determine the right error threshold for rollback?"},
            ]),
            ("API design: REST vs GraphQL", [
                {"speaker": "other", "text": "Our frontend team is pushing for GraphQL, but backend wants to stay REST."},
                {"speaker": "user", "text": "What's the relationship between frontend and backend teams?"},
                {"speaker": "other", "text": "Separate teams with different managers."},
                {"speaker": "user", "text": "GraphQL makes sense when frontend needs flexibility and teams are decoupled."},
                {"speaker": "other", "text": "But won't it complicate our caching strategy?"},
            ]),
            ("code review: performance concern", [
                {"speaker": "other", "text": "I left a comment on your PR about the n+1 query issue in the users endpoint."},
                {"speaker": "user", "text": "I saw that, I was planning to fix it with eager loading."},
                {"speaker": "other", "text": "Eager loading will work, but you might also want to add pagination."},
                {"speaker": "user", "text": "Good point, I'll add cursor-based pagination."},
                {"speaker": "other", "text": "And consider adding a database index on the query columns too."},
            ]),
            ("code review: security concern", [
                {"speaker": "other", "text": "I noticed your API endpoint accepts user input directly in the SQL query."},
                {"speaker": "user", "text": "It's behind an ORM, so it should be parameterized automatically."},
                {"speaker": "other", "text": "Are you sure the ORM handles all edge cases? I've seen raw SQL slip through."},
                {"speaker": "user", "text": "Let me double check and add a test for SQL injection."},
            ]),
            ("technical debt discussion", [
                {"speaker": "other", "text": "The codebase is getting unwieldy. We need to address technical debt."},
                {"speaker": "user", "text": "What areas are causing the most friction?"},
                {"speaker": "other", "text": "The test suite is flaky and the build takes forever."},
                {"speaker": "user", "text": "We should fix the flaky tests first, then tackle build time."},
                {"speaker": "other", "text": "Fair. How do we prioritize which flaky tests to fix?"},
            ]),
            ("decision: build vs buy", [
                {"speaker": "other", "text": "Should we build our own ML platform or buy a third-party solution?"},
                {"speaker": "user", "text": "What's your core differentiator as a company?"},
                {"speaker": "other", "text": "We differentiate on model accuracy, not infrastructure."},
                {"speaker": "user", "text": "Then buy. Building a platform is expensive and it's not your competitive advantage."},
                {"speaker": "other", "text": "But we'd lose control over customization."},
            ]),
            ("monitoring strategy", [
                {"speaker": "other", "text": "We have no monitoring on our ML models in production. Where do we start?"},
                {"speaker": "user", "text": "Start with prediction distribution drift and data quality metrics."},
                {"speaker": "other", "text": "And if we see drift, what's the remediation process?"},
                {"speaker": "user", "text": "Automated alerts triggering retraining pipelines."},
            ]),
            ("team: on-call rotation", [
                {"speaker": "other", "text": "Our team hates on-call. How do other teams make it less painful?"},
                {"speaker": "user", "text": "Good runbooks, blameless postmortems, and proper alerting thresholds."},
                {"speaker": "other", "text": "We have those but people still dread being paged at 3 AM."},
                {"speaker": "user", "text": "You might need to reduce alert noise or automate common fixes."},
            ]),
            ("accessibility consideration", [
                {"speaker": "other", "text": "We need to make our app accessible but the design system wasn't built for it."},
                {"speaker": "user", "text": "Start with the most critical user flows and audit them."},
                {"speaker": "other", "text": "Would automated testing catch enough issues, or do we need manual review?"},
                {"speaker": "user", "text": "Automated tools catch about 30% of issues. You need both."},
            ]),
            ("cross-team dependency", [
                {"speaker": "other", "text": "We're blocked on the data team's API which is delayed by two weeks."},
                {"speaker": "user", "text": "Can you work with mocked data in the meantime?"},
                {"speaker": "other", "text": "We could, but we won't know if it works until integration."},
                {"speaker": "user", "text": "Define the contract upfront and test against it with fake data."},
            ]),
            ("tech stack migration", [
                {"speaker": "other", "text": "We're thinking of migrating from Java to Kotlin. Worth it?"},
                {"speaker": "user", "text": "Kotlin is cleaner but interop can be tricky with existing Java code."},
                {"speaker": "other", "text": "We'd do it gradually, file by file."},
                {"speaker": "user", "text": "That's the safest approach, but it'll take discipline to avoid half-migrated files."},
            ]),
            ("testing philosophy", [
                {"speaker": "other", "text": "Our team doesn't write unit tests, only integration tests. Good enough?"},
                {"speaker": "user", "text": "Integration tests catch different things. You want both."},
                {"speaker": "other", "text": "But maintaining both slows us down significantly."},
                {"speaker": "user", "text": "Focus unit tests on complex logic, integration tests on workflows."},
            ]),
            ("disaster recovery planning", [
                {"speaker": "other", "text": "We had an outage last week and realized our backup strategy is a mess."},
                {"speaker": "user", "text": "What's your RTO and RPO for critical services?"},
                {"speaker": "other", "text": "We haven't defined them formally."},
                {"speaker": "user", "text": "That's the first thing to do before designing backup strategy."},
            ]),
            ("open source contribution strategy", [
                {"speaker": "other", "text": "Our company wants to start contributing to open source. Where do we start?"},
                {"speaker": "user", "text": "Pick projects your team already uses and start small."},
                {"speaker": "other", "text": "Legal is worried about IP contamination though."},
                {"speaker": "user", "text": "Have a clear contribution policy and CLA in place before anyone writes code."},
            ]),
            ("interviewing as an IC", [
                {"speaker": "other", "text": "We need to improve our engineering interview process. Any ideas?"},
                {"speaker": "user", "text": "Make it practical — a real coding problem, not a whiteboard puzzle."},
                {"speaker": "other", "text": "Some interviewers prefer algorithm questions though."},
                {"speaker": "user", "text": "You can still test algorithms, just make it relevant to the work they'll do."},
            ]),
            ("retrospective improvement", [
                {"speaker": "other", "text": "Our retrospectives are boring and nobody speaks up."},
                {"speaker": "user", "text": "Try different formats — start, stop, continue; or anonymous voting."},
                {"speaker": "other", "text": "We tried anonymous but people still don't engage."},
                {"speaker": "user", "text": "Maybe the issue is psychological safety, not format."},
            ]),
            ("team velocity discussion", [
                {"speaker": "other", "text": "Our velocity has been dropping for three sprints. What could be wrong?"},
                {"speaker": "user", "text": "Check if scope has been increasing or if there's technical debt."},
                {"speaker": "other", "text": "Scope has been pretty stable, but complexity has gone up."},
                {"speaker": "user", "text": "You might need to refactor the problematic area or split it into smaller stories."},
            ]),
            ("career growth conversation", [
                {"speaker": "other", "text": "I've been a senior engineer for 4 years. What's the next step if I don't want to manage?"},
                {"speaker": "user", "text": "Look into staff engineer roles or principal IC tracks."},
                {"speaker": "other", "text": "How do staff engineers differ from senior in practice?"},
                {"speaker": "user", "text": "More cross-team influence, technical strategy, and mentoring."},
            ]),
        ])
    ],

    # ── Category D: Casual conversation (25 conversations) ────────────────
    *[
        {"id": f"D{i}", "category": "casual", "label": label, "turns": turns}
        for i, (label, turns) in enumerate([
            ("weekend plans", [
                {"speaker": "other", "text": "Any plans for the weekend?"},
                {"speaker": "user", "text": "Thinking of hiking if the weather holds up."},
                {"speaker": "other", "text": "Oh nice, which trail?"},
                {"speaker": "user", "text": "Not sure yet, maybe the one near the lake you mentioned."},
                {"speaker": "other", "text": "That one's great, the view from the top is amazing."},
            ]),
            ("hobby: gaming", [
                {"speaker": "other", "text": "Have you played the new Zelda yet?"},
                {"speaker": "user", "text": "Not yet, I'm still working through the previous one."},
                {"speaker": "other", "text": "It's so good, you need to prioritize it."},
                {"speaker": "user", "text": "I barely have time these days."},
                {"speaker": "other", "text": "Tell me about it. Adult life ruins gaming."},
            ]),
            ("food recommendations", [
                {"speaker": "other", "text": "I'm craving sushi, any good places nearby?"},
                {"speaker": "user", "text": "There's a great place on 5th street, small family-run."},
                {"speaker": "other", "text": "Is the fish fresh? I'm picky about that."},
                {"speaker": "user", "text": "Super fresh, they get deliveries daily."},
                {"speaker": "other", "text": "Nice, want to go there for lunch tomorrow?"},
            ]),
            ("travel stories", [
                {"speaker": "other", "text": "I just got back from Vietnam. It was incredible."},
                {"speaker": "user", "text": "Oh awesome! Where did you go?"},
                {"speaker": "other", "text": "Hanoi, Ha Long Bay, and Hoi An."},
                {"speaker": "user", "text": "Ha Long Bay is on my bucket list."},
                {"speaker": "other", "text": "You have to go, it's even better than the photos."},
            ]),
            ("movie discussion", [
                {"speaker": "other", "text": "Have you seen Dune Part Two yet?"},
                {"speaker": "user", "text": "Not yet, trying to avoid spoilers."},
                {"speaker": "other", "text": "You should go this weekend, the visuals are stunning."},
                {"speaker": "user", "text": "I heard the sound design is incredible too."},
                {"speaker": "other", "text": "Yeah, best sound mixing I've heard in years."},
            ]),
            ("weather complaint", [
                {"speaker": "other", "text": "This weather is killing me. When does it end?"},
                {"speaker": "user", "text": "Forecast says another week of rain."},
                {"speaker": "other", "text": "Ugh. At least it's not snow, I guess."},
                {"speaker": "user", "text": "That's the spirit. Silver linings."},
            ]),
            ("pets conversation", [
                {"speaker": "other", "text": "We just adopted a rescue dog. He's a handful."},
                {"speaker": "user", "text": "Oh congrats! What breed?"},
                {"speaker": "other", "text": "Lab mix, about a year old, still very energetic."},
                {"speaker": "user", "text": "They calm down after about two years. Good luck!"},
                {"speaker": "other", "text": "That's what everyone says. I'm surviving on coffee."},
            ]),
            ("fitness chat", [
                {"speaker": "other", "text": "I joined a new gym, trying to get back in shape."},
                {"speaker": "user", "text": "Nice! What kind of workouts are you doing?"},
                {"speaker": "other", "text": "Mix of weights and cardio, but I'm so sore I can barely walk."},
                {"speaker": "user", "text": "That passes after the first couple weeks."},
                {"speaker": "other", "text": "Hope so. I forgot how much starting from zero hurts."},
            ]),
            ("book recommendation", [
                {"speaker": "other", "text": "I just finished Project Hail Mary by Andy Weir. You should read it."},
                {"speaker": "user", "text": "Oh I loved The Martian. Is it similar?"},
                {"speaker": "other", "text": "Same vibe but with more humor. And the science is fascinating."},
                {"speaker": "user", "text": "Alright, adding it to my list."},
                {"speaker": "other", "text": "Trust me, you won't regret it."},
            ]),
            ("music exchange", [
                {"speaker": "other", "text": "What have you been listening to lately?"},
                {"speaker": "user", "text": "Mostly lo-fi hip hop, good for focusing."},
                {"speaker": "other", "text": "You should check out the new Glass Animals album if you like chill vibes."},
                {"speaker": "user", "text": "Oh I didn't know they dropped something new!"},
                {"speaker": "other", "text": "Yeah came out last month, it's amazing."},
            ]),
            ("traffic complaint", [
                {"speaker": "other", "text": "Took me two hours to get here. The traffic was insane."},
                {"speaker": "user", "text": "Everyone said the new highway would help but it made things worse."},
                {"speaker": "other", "text": "Yeah, I don't know why they thought adding one lane would fix it."},
                {"speaker": "user", "text": "Induced demand. More lanes just mean more cars."},
            ]),
            ("apartment hunting", [
                {"speaker": "other", "text": "I'm looking for a new apartment and the prices are crazy right now."},
                {"speaker": "user", "text": "Tell me about it. What area are you looking at?"},
                {"speaker": "other", "text": "Anywhere within 30 minutes of downtown, but that rules out anything affordable."},
                {"speaker": "user", "text": "Have you looked at the east side? It's getting better."},
            ]),
            ("technology frustration", [
                {"speaker": "other", "text": "My laptop just blue-screened in the middle of a meeting. So embarrassing."},
                {"speaker": "user", "text": "Classic Windows moment. What happened?"},
                {"speaker": "other", "text": "No idea, it just crashed. I lost all my notes."},
                {"speaker": "user", "text": "That's why I switched to auto-saving cloud notes."},
            ]),
            ("parenting chat", [
                {"speaker": "other", "text": "My kid just started kindergarten and it's harder on me than on him."},
                {"speaker": "user", "text": "I can imagine. How's he adjusting?"},
                {"speaker": "other", "text": "He loves it, runs in every morning without looking back."},
                {"speaker": "user", "text": "That's the dream, right?"},
                {"speaker": "other", "text": "I know, I should be happy but I miss him."},
            ]),
            ("health concern", [
                {"speaker": "other", "text": "I've been having these headaches and I can't figure out why."},
                {"speaker": "user", "text": "Could be eye strain if you're looking at screens all day."},
                {"speaker": "other", "text": "I do spend like 10 hours a day coding."},
                {"speaker": "user", "text": "Try the 20-20-20 rule, look away every 20 minutes."},
            ]),
            ("productivity struggle", [
                {"speaker": "other", "text": "I've been super unproductive this week. Can't focus on anything."},
                {"speaker": "user", "text": "Same here. Maybe it's the weather change."},
                {"speaker": "other", "text": "Or maybe I just need a break. When's your next day off?"},
                {"speaker": "user", "text": "Friday. Counting down the hours."},
            ]),
            ("dinner party planning", [
                {"speaker": "other", "text": "We're hosting a dinner party next Saturday. Want to come?"},
                {"speaker": "user", "text": "Love to! What should I bring?"},
                {"speaker": "other", "text": "Just yourself and maybe a bottle of wine if you drink."},
                {"speaker": "user", "text": "Perfect, I'll bring a nice red."},
                {"speaker": "other", "text": "Great, starts around 7!"},
            ]),
            ("home renovation story", [
                {"speaker": "other", "text": "We started renovating the kitchen and found mold behind the cabinets."},
                {"speaker": "user", "text": "Oh no, that's the worst. How bad is it?"},
                {"speaker": "other", "text": "Pretty extensive. Contractor says it'll add two weeks and 5 grand."},
                {"speaker": "user", "text": "At least you found it now instead of after finishing everything."},
            ]),
            ("funny story share", [
                {"speaker": "other", "text": "You won't believe what happened at the office today."},
                {"speaker": "user", "text": "What happened?"},
                {"speaker": "other", "text": "The CEO accidentally joined a standup thinking it was a board meeting."},
                {"speaker": "user", "text": "Oh no, that's hilarious. What did everyone do?"},
                {"speaker": "other", "text": "Just sat there awkwardly while he talked about quarterly targets."},
            ]),
            ("phone upgrade dilemma", [
                {"speaker": "other", "text": "My phone is 4 years old and barely holding up. Time to upgrade?"},
                {"speaker": "user", "text": "If it still works, keep it. New phones aren't that different."},
                {"speaker": "other", "text": "The battery lasts like 3 hours though."},
                {"speaker": "user", "text": "Okay that's a legit reason. Get a battery replacement first."},
            ]),
            ("cooking attempt", [
                {"speaker": "other", "text": "I tried to make sourdough bread and it came out like a rock."},
                {"speaker": "user", "text": "Sourdough is tricky. Did you feed the starter enough?"},
                {"speaker": "other", "text": "I think that was the problem. It didn't rise at all."},
                {"speaker": "user", "text": "Keep the starter in a warm spot and feed it daily for a week."},
            ]),
            ("parking nightmare", [
                {"speaker": "other", "text": "I spent 30 minutes looking for parking. Ridiculous."},
                {"speaker": "user", "text": "Downtown parking is a scam. I just take transit now."},
                {"speaker": "other", "text": "I would, but transit takes an hour each way."},
                {"speaker": "user", "text": "Yeah that's rough. Maybe look into a parking subscription?"},
            ]),
            ("family visit story", [
                {"speaker": "other", "text": "My parents are visiting next month and I'm already stressed."},
                {"speaker": "user", "text": "How long are they staying?"},
                {"speaker": "other", "text": "Two weeks. I love them but that's a lot."},
                {"speaker": "user", "text": "Plan some activities and also schedule alone time. It helps."},
            ]),
            ("new hobby discovery", [
                {"speaker": "other", "text": "I started learning the piano. It's harder than I expected."},
                {"speaker": "user", "text": "It gets easier. How long have you been practicing?"},
                {"speaker": "other", "text": "About two weeks. My fingers don't do what I want them to."},
                {"speaker": "user", "text": "That's normal. Muscle memory takes time."},
            ]),
        ])
    ],
]
