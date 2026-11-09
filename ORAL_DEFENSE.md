# Oral Defense — short English answers and Egyptian Arabic

These answers describe the implemented code. They do not claim that real participants have been verified. The real dataset, fitted biometric models, and measured biometric performance are still pending. Tests with diagnostic patterns and branch doubles are software checks only.

الكلام ده بيشرح الكود اللي اتعمل. لسه لازم نجمع داتا حقيقية وندرّب ونقيّم. نجاح اختبارات الكود مش معناه إننا جرّبنا التعرف على ناس حقيقية.

## Explain the whole project in one minute

**English:** A person chooses a claimed user ID. We check a live face, one spoken command's speaker, the concatenated face/voice features, and a new ink fingerprint impression, in that order. Each classifier prediction must match the claimed ID and pass a distance threshold. Only then do we recognize the word from the same audio features, check the verified user's role, and print an authorized simulated action. A failure stops everything after that stage.

**مصري:** الشخص بيختار هو بيدّعي إنه مين. بنفحص الوش، وبعدين صوت كلمة واحدة، وبعدين خصائص الوش والصوت مع بعض، وبعدين بصمة حبر جديدة. كل مرحلة لازم تقول نفس الهوية المطلوبة والمسافة تكون تحت الحد. بعد كده بس بنحدد الكلمة ونشوف صلاحية الشخص ونطبع نتيجة تجريبية. أول فشل بيوقف المحاولة.

## Every module: what, why, input, output, callers, and concept

### `config.py`

**English:** Stores paths, enrolled IDs/roles, four commands, feature settings, and the session split rule. We need one place to edit the team's setup. Input: constants edited by students and a session ID for `session_split`. Output: settings and TRAIN/VALIDATION/TEST. Called by all application modules; those modules receive its settings. Concept: fixed experimental setup and role dictionaries, not a learned model.

**مصري:** ده مكان الإعدادات: المستخدمين والأدوار والمسارات والكلمات وتقسيم الجلسات. بندخل الإعدادات ورقم الجلسة، وباقي الملفات بتاخدها منه. الهدف كلنا نستخدم نفس النظام، ومفيش تعلم آلي في الملف ده.

### `collect_data.py`

**English:** Captures actual camera images, records two seconds of audio, imports/captures a new ink impression, and probes devices. We need genuine team samples and live attempts. Input: participant ID, session, camera/microphone, optional new image path. Output: raw files and a rebuilt manifest, or live arrays/availability information. Called by `main` and `pipeline`; files go to `dataset`/training, and live arrays go to `features`. Concept: data acquisition and basic signal/image input; no identity prediction.

**مصري:** بيجمع صور وصوت حقيقيين وبصمات حبر، وبيشوف الأجهزة متاحة ولا لأ. بيدخله المستخدم والجلسة والأجهزة أو صورة بصمة جديدة. بيطلع ملفات للداتا أو مصفوفات للفحص المباشر. القائمة والبايبلاين بينادوه، والخصائص بتتطلع بعده. ده جمع داتا، مش موديل تعرف.

### `dataset.py`

**English:** Loads/rebuilds the CSV and validates counts, ownership paths, missing/duplicate files, and session splits. We need to reject inconsistent experiments before training. Input: manifest plus raw files. Output: rows, counts, errors, or a dataset signature. Called by collection, main, training, and evaluation; training/evaluation receive validated rows. Concept: session-based holdout and leakage prevention. A filename check can detect metadata mismatch, but cannot prove whose real face or voice a file contains.

**مصري:** بيقرأ جدول الداتا وبيشيك العدد والمسارات والملفات الناقصة والتكرار والتقسيم. بيدخله الجدول والملفات وبيطلع صفوف سليمة أو أخطاء. التدريب والتقييم بيعتمدوا عليه عشان الداتا متتسربش بين التقسيمات. اسم الملف يساعدنا نكشف غلط في التسجيل، لكنه مش دليل إن الشخص الحقيقي في الصورة هو صاحب الاسم.

### `features.py`

**English:** Converts raw images/audio into fixed numeric vectors. We need numbers of consistent length for KNN/SVM. Input: image, waveform, file path, or face/voice vectors with TRAIN scalers. Output: face 4366 values, voice 30, fingerprint 8110, or fusion 4396. Called by collector, trainer, pipeline, and individual demos; vectors go to training/verifiers/command recognition. Concepts: fixed crops, grayscale/resize, HOG, LBP, MFCC/ZCR/RMS statistics, normalization, and concatenation.

**مصري:** بيحوّل الصورة والصوت لأرقام ثابتة العدد عشان الموديل يقدر يستخدمها. الوش 4366 رقم، والصوت 30، والبصمة 8110، والدمج 4396. الجمع والتدريب والبايبلاين بينادوه. الناتج بيروح للموديلات. أهم حاجة نفس الخطوات بالضبط وقت التدريب والتجربة.

### `train_models.py`

**English:** Fits TRAIN-only scalers, KNN(k=3), and linear SVM for five tasks. We need real models, TRAIN mean templates, and defensible rejection thresholds. Input: a complete validated custom dataset. Output: five joblib bundles containing ten named models, scaler/template/label information, selected names, thresholds, and validation evidence. Called by main or the direct training script; saved bundles go to verification/commands/evaluation. Concepts: supervised classification, normalization, validation macro F1 selection, and Euclidean-distance threshold calibration. After saving, it calls frozen TEST evaluation.

**مصري:** بيدرّب KNN وSVM لكل مهمة، وبيتعلم التطبيع ومتوسط خصائص كل هوية من التدريب بس. بيدخله داتا حقيقية كاملة وبيطلع الموديلات والإعدادات والحدود. التحقق والتعرف على الأوامر والتقييم بيستخدموا الناتج. بنختار الموديل من التحقق مش من الاختبار، وبعد الحفظ بنقيّم من غير تعديل.

### `verification.py`

**English:** Checks a claimed identity using the selected classifier and a distance gate. We need verification rather than blindly accepting the closest class. Input: biometric vector, claimed ID, and local saved bundle. Output: accepted, predicted/claimed IDs, distance, threshold, and reason. Called by pipeline, individual demos, and evaluation; decisions return to those callers. Concept: identification prediction plus claimed-user template verification. Missing/corrupt model state rejects.

**مصري:** بياخد خصائص العينة والهوية اللي الشخص بيدّعيها. لازم توقع الموديل يطابق الهوية، وكمان المسافة لمتوسط خصائصها تكون مقبولة. بيطلع قبول أو رفض والسبب والمسافة. البايبلاين بيستخدم القرار عشان يكمل أو يقف. لو موديل أو إعداد ناقص، بيرفض.

### `commands.py`

**English:** Predicts which of the four words was spoken, using its own command model and distance rejection. We need to know the requested action separately from who spoke. Input: the same 30-dimensional vector already used for voice verification. Output: accepted command or UNKNOWN, distance, threshold, and reason. Called by pipeline after every biometric gate passes; its output goes back to pipeline for permission checking. Concepts: supervised word classification and template-distance rejection. It executes no action.

**مصري:** ده بيحدد الكلمة مش صاحب الصوت. بياخد نفس الثلاثين رقم اللي اتستخدموا للتحقق من المتكلم، ويطلع الكلمة أو UNKNOWN. البايبلاين بيناديه بعد نجاح كل فحوص الهوية، وبعده بنفحص الصلاحيات. الملف نفسه مينفذش أي أمر.

### `permissions.py`

**English:** Looks up a user's role and whether it allows the word. We need authorization after authentication. Input: verified user ID and command. Output: role/Boolean permission. Called by pipeline; the decision returns to pipeline. Concept: dictionaries and conditionals, not ML. The caller must establish verified identity before asking permission.

**مصري:** بيشوف دور المستخدم المتحقق منه وهل الأمر مسموح له. بيدخله الهوية والكلمة وبيطلع الدور ومسموح ولا لأ. البايبلاين بيناديه بعد التحقق. ده قاموس وشروط عادية، مش موديل بيخمن الصلاحيات.

### `actions.py`

**English:** Prints the authorized simulated STATUS, REPORT, UNLOCK, or LOGOUT response. We need a harmless visible outcome. Input: authorized supported command. Output: printed text and a result dictionary with a session-end flag. Called only by the live pipeline after authorization; result returns to pipeline and then the CLI. Concept: ordinary conditional/dictionary dispatch; no ML and no real lock or OS access. LOGOUT ends that attempt.

**مصري:** بيطبع نتيجة تجريبية للأمر المسموح. البايبلاين بس بيناديه بعد التحقق والصلاحيات. مفيش قفل حقيقي ولا أوامر نظام. الناتج رسالة وعلامة لو الجلسة خلصت. LOGOUT بيقفل المحاولة دي.

### `pipeline.py`

**English:** Connects the ordered live workflow and returns all decisions. We need one actual protected path. Input: claimed ID and actual captures, or explicit held-out samples for a no-action evaluation. Output: decisions, verified ID, word, role, permission, action/rejection stage and reason. Called by main and evaluation; result goes to display or trial metrics. Concepts: hierarchical conditional verification, early feature fusion, and authorization. Offline supplied samples never execute actions.

**مصري:** ده اللي بيربط المشروع كله بالترتيب. بياخد الهوية والعينات ويشغّل المراحل واحدة واحدة. بيطلع كل القرارات وسبب الرفض أو النتيجة. القائمة والتقييم بينادوه. أول رفض بيوقف باقي المراحل، وتجارب الملفات المحفوظة عمرها ما بتنفذ فعل حتى لو اتقبلت.

### `evaluate.py`

**English:** Compares both algorithms on TEST and evaluates genuine/wrong-identity claims through selected verifiers and the real hierarchy. We need measured results without tuning on TEST. Input: frozen artifacts and genuine held-out sessions 09–10. Output: metric CSVs, confusion-matrix PNGs, trial CSV, and setup JSON; it executes no actions. Called by trainer, main, or its direct script; students/readers consume its reports. Concepts: accuracy, macro precision/recall/F1, confusion matrix, FAR, and FRR. It also explains fingerprint reuse across correlated trials.

**مصري:** بيقارن الموديلات على جلسات الاختبار بس، وبيجرب ادعاء الهوية الصح والهويات الغلط من غير تغيير العينة. بيطلع جداول وصور ومعلومات التجربة. التدريب والقائمة يقدروا ينادوه. مفيش تعديل للموديل من نتيجة الاختبار ومفيش تنفيذ أوامر. نفس بصمة الجلسة بتتكرر مع الأربع كلمات، فلازم نقول إن التجارب دي مش مستقلة تمامًا.

### `main.py`

**English:** Provides the ten-option command-line menu. We need one simple student-facing entry point. Input: menu choices, claimed user/session, and optional new fingerprint path. Output: module calls and printed results. Called by the student; it calls the appropriate collection/training/evaluation/verification/pipeline module, and the student receives the displayed result. Concept: straightforward application control flow, not a new ML algorithm. Only option 9 reaches the protected action pipeline.

**مصري:** دي القائمة الرئيسية. الطالب بيختار منها جمع أو تدريب أو تقييم أو تجربة. بتاخد اختياره وبتنادي الملف المناسب وتعرض النتيجة. مفيش تعلم آلي جديد هنا. تجارب الوسائل المنفردة مبتنفذش أوامر؛ الاختيار 9 هو المسار الكامل.

### `tests/`

**English:** Checks features, permissions, pipeline stops, and dataset integrity. We need evidence that the software branches correctly. Input: diagnostic checkerboards/tones/numbers, temporary integrity markers, and simple pipeline doubles. Output: unittest pass/fail. Called by unittest; developers receive the report. Concept: software testing, not biometric evaluation. We do not use these fixtures to claim a trained biometric model or real accuracy.

**مصري:** دي اختبارات للكود: الأبعاد والصلاحيات والتوقف عند الرفض وسلامة الجدول. بنستخدم أنماط وأرقام تجريبية وبدائل بسيطة لقرارات البايبلاين. الناتج نجاح أو فشل اختبار، مش نسبة تعرف على بشر. داتا الاختبار البرمجي مش داتا التسجيل الحقيقية.

## Common questions

### Verification versus identification

**English:** Identification asks “Which enrolled user is this?” Verification asks “Does this sample support the specific claimed ID?” Our classifier gives an identity, but acceptance also needs claim agreement and a template-distance gate.

**مصري:** التعرف على الهوية بيختار مين من المستخدمين. التحقق بيسأل هل الشخص ده فعلًا هو الهوية اللي بيدّعيها. مجرد أقرب اسم مش كفاية للقبول.

### Recognition versus verification

**English:** Recognition assigns a label, such as a word. Verification tests a claim and can reject it. Word recognition does not prove speaker identity.

**مصري:** التعرف بيطلع تصنيف زي اسم كلمة. التحقق بيفحص ادعاء وممكن يرفضه. معرفة إن الكلمة UNLOCK مش معناها إننا عرفنا مين قالها.

### Authentication versus authorization

**English:** Authentication verifies who the user is. Authorization checks what the verified user may do. A verified Employee still cannot UNLOCK.

**مصري:** التحقق من الهوية معناه أنت مين. الصلاحيات معناها مسموح لك تعمل إيه بعد ما عرفناك. الموظف ممكن يتعرف صح ويتمنع من فتح الوصول.

### KNN versus SVM

**English:** KNN(k=3) votes among the three nearest training examples. Linear SVM learns linear class boundaries with a separating margin. Both use the same standardized task features, and validation decides which model we select.

**مصري:** KNN بيبص لأقرب تلات أمثلة ويختار بتصويتهم. SVM بيتعلم حدود فصل خطية بين الفئات. بنستخدم نفس الخصائص المطَبّعة وبنقارن على داتا التحقق.

### HOG versus LBP

**English:** HOG summarizes directions of intensity changes, which describe local shape/edges. LBP compares each pixel with neighbors and summarizes local texture. We concatenate both rather than using a neural image model.

**مصري:** HOG بيلخص اتجاهات التغير في الصورة والحواف. LBP بيقارن البكسل بجيرانه عشان يوصف الملمس. بنضم الوصفين لبعض.

### What are MFCC, ZCR, and RMS?

**English:** MFCC describes the audio's short-time spectral shape. ZCR counts sign changes, and RMS measures signal energy. Means and standard deviations across frames produce 13+13+1+1+1+1 = 30 numbers.

**مصري:** MFCC بيوصف شكل طيف الصوت في فترات صغيرة. ZCR معدل عبور الإشارة للصفر، وRMS طاقة الصوت. المتوسط والانحراف المعياري بيدونا 30 قيمة ثابتة مهما كان عدد الفريمات.

### Why early fusion? How is it different from late fusion?

**English:** Early fusion lets a separate model learn from both modalities' combined numeric features. We scale face and voice separately and concatenate before classification. Late fusion would combine predictions/scores after separate classifiers; we do not implement that. Early fusion may help but is not guaranteed to win.

**مصري:** بنجمع خصائص الوش والصوت قبل الموديل عشان يتعلم منهم مع بعض. الدمج المتأخر بيجمع القرارات بعد الموديلات. إحنا عندنا خصائص متجمعة وموديل مستقل، ومش بنفترض إنه لازم يبقى أحسن.

### Why normalization? Why fit it only on TRAIN?

**English:** Features use different numeric scales; StandardScaler subtracts TRAIN means and divides by TRAIN standard deviations. It prevents large units alone from dominating distance. Validation/test statistics must not influence the fitted transformation. Scaling dimensions does not equalize face and voice block lengths.

**مصري:** الأرقام مقاييسها مختلفة، فبنطرح متوسط التدريب ونقسم على انحرافه. كده الوحدة الكبيرة متكسبش لمجرد حجمها. لو اتعلمنا التطبيع من الاختبار يبقى سربنا معلومات. وبرضه عدد خصائص الوش الأكبر ممكن يخليه مؤثر أكتر.

### Why distance thresholds? How are they selected?

**English:** A classifier tends to pick an enrolled label even for unfamiliar input. We also require closeness to the claimed user's TRAIN mean. On VALIDATION only, try zero and observed distances; minimize average FAR/FRR, then prefer lower FAR and a smaller threshold on ties. Never tune this on TEST.

**مصري:** الموديل ممكن يختار اسم معروف لأي حاجة غريبة. عشان كده لازم العينة تبقى قريبة من متوسط خصائص الهوية المطلوبة. بنختار الحد من التحقق بتوازن الرفض والقبول الغلط، ومع التعادل نفضل قبول غلط أقل. الاختبار مش للاختيار.

### What is FAR?

**English:** False Acceptance Rate is `FP / (FP + TN)`: the fraction of impostor claims accepted. Lower is better. A zero denominator is undefined, not evidence of zero false accepts.

**مصري:** دي نسبة ادعاءات الهوية الغلط اللي اتقبلت من كل الادعاءات الغلط. لو مفيش تجارب من النوع ده، منقدرش نقول النسبة صفر؛ هي مش محسوبة.

### What is FRR?

**English:** False Rejection Rate is `FN / (FN + TP)`: the fraction of genuine claims rejected. Lower is better. Tightening a distance threshold often reduces FAR while increasing FRR.

**مصري:** دي نسبة أصحاب الهوية الحقيقيين اللي اترفضوا. تقليل الحد غالبًا بيقلل قبول الغرباء، لكن ممكن يرفض ناس حقيقية أكتر.

### Why separate TRAIN, VALIDATION, and TEST by session?

**English:** TRAIN learns scalers, models, and templates. VALIDATION selects models and thresholds. TEST estimates final performance without updates. Session grouping prevents closely related samples from one acquisition session leaking across splits. Fingerprint references within one session are not independent samples.

**مصري:** التدريب للتعلم، والتحقق للاختيار، والاختبار للقياس النهائي. بنقسم بالجلسات عشان أمثلة متشابهة من نفس الوقت متدخلش الناحيتين وتدي نتيجة مضللة. تكرار مرجع البصمة مش بصمات جديدة.

### Why must all users record the same four words?

**English:** The word model must learn words across speakers, while the identity model must learn speakers across words. Otherwise a word or permission could accidentally reveal identity.

**مصري:** عايزين موديل الكلمة يشوف كل الناس بتقولها، وموديل الشخص يشوفه بيقول كل الكلمات. لو كل شخص قال أوامر مختلفة، الموديل ممكن يتعلم الكلمة بدل صاحب الصوت.

### Why reuse the same audio for verification and word recognition?

**English:** It binds the verified speaker to the requested word in this attempt and avoids a different recording being substituted later. We record once, extract once, and pass the same vector to both separate models.

**مصري:** عشان نفس التسجيل اللي وثقنا صاحبه هو اللي نعرف منه الطلب. منسجلش مرة تانية ممكن تخص حد تاني. التسجيل واحد والخصائص واحدة، لكن الموديلين وأهدافهم مختلفين.

### Why verify before recognizing the command?

**English:** The protected workflow requires identity first. A failed identity must not reach command recognition or action execution. Recognizing a word alone never grants permission.

**مصري:** المسار المطلوب يبدأ بالهوية. لو أي فحص هوية فشل، منوصلش لتحديد الأمر أو تنفيذه. معرفة الكلمة لوحدها مش تصريح.

### Why can a verified person still receive ACCESS DENIED?

**English:** Their role may forbid the recognized command. For example, Guest can STATUS/LOGOUT but cannot REPORT. An unknown or unreliable command is rejected separately as well.

**مصري:** لأن الهوية والصلاحية حاجتين مختلفتين. الضيف معروف بس التقرير مش من صلاحياته. وكمان لو الكلمة مش موثوقة ممكن تترفض حتى مع نجاح الهوية.

### What happens if one verification module fails?

**English:** Stop immediately. Do not capture later inputs, recognize a command, or execute an action. Return the failing stage and reason. Missing models/scalers/thresholds and invalid input also reject.

**مصري:** بنقف فورًا ونوضح أنهي مرحلة فشلت وليه. منكمّلش جمع المراحل اللي بعدها ولا ننفذ أمر. نقص الموديل أو بيانات بايظة كمان معناه رفض.

### How are the four bonuses integrated?

**English:** Custom collection supplies matched face/audio and real ink impressions. Live capture supplies new attempts. The face→voice→fusion→fingerprint chain authenticates the claim. The command model reuses the voice vector, then role checking protects the simulated action. They share actual data and decisions rather than being four separate demonstrations.

**مصري:** الداتا المخصصة بتوفر عينات متطابقة، والتصوير والتسجيل المباشر بيجيبوا محاولة جديدة، والبصمة جزء إجباري من التحقق. بعد نجاح السلسلة بنحدد الأمر من نفس الصوت ونراجع الصلاحيات. كله متوصل ببيانات وقرارات حقيقية.

### How do we evaluate the whole hierarchy fairly?

**English:** Use matched TEST user/session/word rows, claiming the correct user and each wrong enrolled user while leaving samples unchanged. Biometric acceptance means all four gates passed. Record word/permission outcomes separately. Reusing a session's fingerprint for four words is disclosed, and no offline actions execute.

**مصري:** بناخد عينة اختبار متطابقة ونغير ادعاء الهوية بس، مش الوش والصوت. النجاح البيومتري معناه الأربع مراحل نجحت. الصلاحيات والكلمة ليهم تسجيل لوحدهم عشان رفض أمر ممنوع ميتحسبش فشل هوية. وبنقول بوضوح إن بصمة الجلسة مستخدمة مع الأربع كلمات.

### Can we claim unknown-word or unknown-person security?

**English:** No guarantee. Distances can reject some unfamiliar inputs, but the small known-user dataset has no representative unknown population or out-of-vocabulary word set. Command calibration uses wrong known labels only as a proxy. We report this limit rather than claiming perfect rejection.

**مصري:** لا، الحد ممكن يرفض بعض العينات الغريبة لكنه مش ضمان. معندناش مجموعة كفاية لناس وكلمات جديدة. مقارنة كلمات معروفة بادعاءات غلط مجرد بديل محدود للتجربة، مش قياس حقيقي لكل المجهول.

### Why is this fingerprint system educational rather than production-grade?

**English:** HOG/LBP describe image appearance, not specialized minutiae matching. Rotation, ink pressure, lighting, and crop can change distances substantially. There is no liveness or spoof protection. Consistent genuine acquisition is essential, and performance must be measured.

**مصري:** إحنا بنوصف شكل الصورة وملمسها، مش بنطابق تفاصيل البصمة المتخصصة. اللفة والضغط والإضاءة والقص بيأثروا جامد. مفيش كشف تزوير أو حيوية، فلازم تصوير ثابت وقياس حقيقي للنتيجة.

### What has actually been tested so far?

**English:** See `results/STATUS.md` for the latest actual run. Software tests cover dimensions, preprocessing, threshold arithmetic, manifest integrity, permissions, safe rejection, and pipeline order/reuse. No real biometric training or performance evaluation has occurred. Camera availability failed; microphone enumeration/settings succeeded, without a real recording test.

**مصري:** اختبرنا الكود والتوقف والصلاحيات وشكل الخصائص، مش دقة على مشاركين حقيقيين. لسه مفيش داتا ولا موديلات بيومترية مدربة. الكاميرا مش متاحة، والميكروفونات ظهرت في الفحص بس مجربناش تسجيل حقيقي.

## Suggested four-person oral presentation

1. Student 1: collection, manifest, splits, data integrity, and genuine fingerprint acquisition.
2. Student 2: HOG/LBP, MFCC/ZCR/RMS, TRAIN normalization, and early fusion.
3. Student 3: KNN/SVM, validation selection, claimed-identity distance gate, FAR/FRR, and TEST evaluation.
4. Student 4: full pipeline, one-recording reuse, commands, roles, simulated actions, and stop tests.

كل واحد يمسك جزء في العرض، لكن لازم كل الفريق يفهم السلسلة كاملة ويعرف يشرح المدخلات والمخرجات وسبب الرفض في كل مرحلة. لو الفريق تلاتة، وزعوا جزء البايبلاين والاختبارات بينهم.
