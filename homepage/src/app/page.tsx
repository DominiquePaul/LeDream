'use client';

import { useState, useEffect } from 'react';
import AsciiEurope from '@/components/AsciiEurope';
import WaitlistCounter from '@/components/WaitlistCounter';

type Language = 'en' | 'de';

const content = {
  en: {
    slogan: "Making Skill Scale",
    oneliner: "Dream Machines lets non-roboticists automate variable, repetitive manufacturing work in SMEs by teaching robots through demonstration.",
    paragraphs: [
      "Europe's manufacturing strength is built on precision, process knowledge, and decades of accumulated expertise. Many companies with just a few hundred employees are global leaders in their niche. Their advantage is not scale, but skill.",
      "That model is reaching its limits. Labour is scarce. Costs are rising. Experienced operators are retiring. Routine work has become a global commodity. If European industry wants to stay competitive, it must automate what does not differentiate it and multiply the know-how that does.",
      "We are building for a world in which automated microfactories can be set up and reconfigured by the people who understand the process, not by teams of robotics specialists. Automation should capture skill and replicate it reliably, so that production scales with expertise rather than with available labour.",
      "Our first product addresses the most immediate constraint: variable, repetitive workbench tasks that are essential to operations but too inconsistent for traditional automation. Dream Machines develops robotic AI, packaged as a full solution that integrates into existing workstations and learns directly from operators.",
      "Instead of programming every movement, your team demonstrates the task. When products or processes change, the automation is updated through new demonstrations rather than long integration projects or mechanical redesigns.",
      "We focus on semi-structured tasks such as PCB testing, battery slotting, machine tending with variable parts, picking and sorting irregular objects, and quality control. These are critical to keeping factories running, yet rarely define strategic advantage and are exactly where intelligent robotics creates leverage.",
      "Dream Machines was founded by researchers from ETH Zürich and is currently preparing its first pilot projects with European manufacturers."
    ],
    contactNote: "For enquiries, reach out to",
    waitlist: {
      headline: "Waitlist",
      subtext: "Join to get priority access to demos",
      placeholder: "Your email",
      cta: "Join waitlist",
      success: "You're on the list.",
      error: "Something went wrong. Please try again."
    }
  },
  de: {
    slogan: "Wir skalieren Routinearbeit",
    oneliner: "Dream Machines ermöglicht es nicht-technischen Fachkräften in KMUs nicht-standardisierte, repetitive Arbeit zu automatisieren, indem sie Robotern Aufgaben durchs Vormachen beibringen.",
    paragraphs: [
      "Die Stärke der europäischen Industrie basiert auf Präzision, Prozesswissen und jahrzehntelang aufgebauter Expertise. Viele Unternehmen sind mit wenigen hundert Mitarbeitenden in ihrer Nische Weltmarktführer. Was sie besonders macht ist nicht ihre Größe, sondern ihre Expertise und Verfahrenstechnik.",
      "Dieses Modell stösst zunehmend an seine Grenzen. Erfahrene Fachkräfte gehen in den Ruhestand. Routinetätigkeiten sind zu global austauschbaren Standardleistungen geworden. Wenn wir in Europa wettbewerbsfähig bleiben wollen, müssen wir das automatisieren, was uns nicht differenziert, und das amplifizieren, was uns einzigartig macht.",
      "Wir bauen für eine Welt, in der Fabriken ohne Menschen von den Menschen eingerichtet und umkonfiguriert werden können, die den Prozess verstehen, nicht von Teams aus Robotikspezialisten. Automatisierung sollte Können erfassen und zuverlässig replizieren, sodass Produktion mit Expertise skaliert und nicht mit verfügbarer Arbeitskraft.",
      "Unser erstes Produkt adressiert die akuteste Einschränkung: variable, repetitive Werkbankaufgaben, die essenziell für viele Betriebe sind, aber zu variabel für klassische Automatisierung. Dream Machines entwickelt eine robotische KI als fertige Lösung, die sich in bestehende Arbeitsplätze ohne Umstellungen integriert und von den eigenen Mitarbeitern konfiguriert werden kann.",
      "Anstatt Bewegungen zu programmieren, demonstrieren Mitarbeiter die Aufgabe mittels Teleoperation durch die Roboterarme selbst. Wenn sich Produkte oder Prozesse ändern, wird die Automatisierung durch neue Demonstrationen aktualisiert statt durch langwierige Neuprogrammierung oder erneute Beauftragung von Systemintegratoren.",
      "Wir ermöglichen die Automatisierung von Aufgaben wie PCB-Tests, das Einsetzen von Batterien, Maschinenbedienung mit variablen Teilen, das Greifen und Sortieren unregelmässiger Objekte sowie allgemeiner Qualitätskontrolle. Diese Tätigkeiten sind entscheidend für den reibungslosen Betrieb von Fabriken, schaffen jedoch selten einen Wettbewerbsvorteil. Genau da ermöglichen wir es mit intelligenter Robotik anzusetzen.",
      "Dream Machines wurde von Forschern der ETH Zürich gegründet und bereitet derzeit die ersten Pilotprojekte mit europäischen Mittelständlern vor."
    ],
    contactNote: "Für Fragen schreiben Sie uns an",
    waitlist: {
      headline: "Warteliste",
      subtext: "Treten Sie bei um bevorzugten Demo Zugang zu erhalten.",
      placeholder: "Ihre Email",
      cta: "Anmelden",
      success: "Sie sind auf der Liste.",
      error: "Etwas ist schiefgelaufen. Bitte erneut versuchen."
    }
  }
};

function detectSystemLanguage(): Language {
  if (typeof window === 'undefined') return 'en';

  const browserLang = navigator.language.toLowerCase();
  if (browserLang.startsWith('de')) {
    return 'de';
  }
  return 'en';
}

export default function Home() {
  const [language, setLanguage] = useState<Language>('en');
  const [mounted, setMounted] = useState(false);
  const [email, setEmail] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  useEffect(() => {
    setMounted(true);

    const savedLang = localStorage.getItem('preferred-language') as Language;
    const initialLang = savedLang || detectSystemLanguage();

    setLanguage(initialLang);
    document.documentElement.lang = initialLang;
  }, []);

  const handleLanguageToggle = () => {
    const newLang = language === 'en' ? 'de' : 'en';
    setLanguage(newLang);
    localStorage.setItem('preferred-language', newLang);
    document.documentElement.lang = newLang;
  };

  if (!mounted) {
    return (
      <div className="min-h-screen">
        <div className="max-w-[1088px] mx-auto px-6">
          <div className="h-6 bg-navy/5 rounded animate-pulse mt-[200px]"></div>
        </div>
      </div>
    );
  }

  const t = content[language];

  return (
    <div className="min-h-screen relative">
      {/* Language toggle */}
      <div className="fixed top-4 right-6 z-50">
        <button
          onClick={handleLanguageToggle}
          className="text-sm text-navy-muted hover:text-navy transition-colors font-dm-mono"
          aria-label={`Switch to ${language === 'en' ? 'German' : 'English'}`}
        >
          {language === 'en' ? 'DE' : 'EN'}
        </button>
      </div>

      {/* ASCII Globe */}
      <div className="max-w-[1088px] mx-auto px-6">
        <AsciiEurope />
      </div>

      {/* Content centered below globe */}
      <div className="max-w-[1088px] mx-auto px-6 flex flex-col items-center">
        {/* Waitlist Counter */}
        {/* <WaitlistCounter language={language} /> */}

        {/* Hero Heading */}
        <div className="text-center mt-10">
          <h1
            className="text-5xl md:text-7xl lg:text-[110px] font-season text-navy"
            style={{ fontVariationSettings: '"wght" 500, "SERF" 70', lineHeight: '0.9', letterSpacing: '-0.01em' }}
          >
            {t.slogan}
          </h1>
        </div>

        {/* Body text */}
        <div className="max-w-[758px] 2xl:max-w-[620px] mt-[70px] text-base lg:text-lg leading-[1.3] font-dm-mono text-navy space-y-6">
          <p>{t.oneliner}</p>
          {t.paragraphs.map((paragraph, index) => (
            <p key={index}>
              {paragraph}
            </p>
          ))}
        </div>

        {/* Contact note */}
        <div className="max-w-[758px] 2xl:max-w-[620px] w-full mt-8">
          <p className="text-navy-muted font-dm-mono text-base lg:text-lg leading-[1.3]">
            {t.contactNote}{' '}
            <a href="mailto:team@dream-machines.eu" className="underline hover:text-navy transition-colors">team@dream-machines.eu</a>
          </p>
        </div>

        {/* Waitlist form */}
        <div className="max-w-[758px] 2xl:max-w-[620px] w-full mt-20 mb-16">
          <h2 className="text-base lg:text-lg leading-[1.3] font-bold mb-1 font-dm-mono text-navy">{t.waitlist.headline}</h2>
          {t.waitlist.subtext && (
            <p className="text-base lg:text-lg leading-[1.3] text-navy-muted mb-3 font-dm-mono">{t.waitlist.subtext}</p>
          )}
          <form
            className="flex flex-col sm:flex-row items-start sm:items-center gap-3"
            onSubmit={async (e) => {
              e.preventDefault();
              setSubmitting(true);
              setMessage(null);
              try {
                const res = await fetch('/api/subscribe', {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ email })
                });
                if (res.ok) {
                  setMessage(t.waitlist.success);
                  setEmail('');
                  window.dispatchEvent(new Event('waitlist-updated'));
                } else {
                  setMessage(t.waitlist.error);
                }
              } catch {
                setMessage(t.waitlist.error);
              } finally {
                setSubmitting(false);
              }
            }}
          >
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder={t.waitlist.placeholder}
              className="w-full sm:w-auto flex-1 border border-navy/20 px-3 py-2 text-base lg:text-lg leading-[1.3] font-dm-mono text-navy bg-transparent focus:outline-none focus:ring-2 focus:ring-navy/30 placeholder:text-navy-muted"
            />
            <button
              type="submit"
              disabled={submitting}
              className="px-5 py-2.5 text-base lg:text-lg leading-[1.3] bg-navy text-cream font-dm-mono font-medium hover:bg-navy-light transition-colors disabled:opacity-60"
            >
              {t.waitlist.cta}
            </button>
          </form>
          {message && (
            <p className="mt-2 text-base lg:text-lg leading-[1.3] text-navy-muted font-dm-mono">{message}</p>
          )}
        </div>
      </div>
    </div>
  );
}
