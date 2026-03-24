'use client';

import { useState, useEffect } from 'react';
import Image from 'next/image';

export default function Angels() {
  const [mounted, setMounted] = useState(false);
  const [password, setPassword] = useState('');
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    setMounted(true);
    document.documentElement.lang = 'en';
    if (sessionStorage.getItem('angels-auth') === '1') {
      setAuthenticated(true);
    }
    // Preload images so they display instantly after password entry
    const img1 = new window.Image();
    img1.src = '/angels-collage.jpg';
    const img2 = new window.Image();
    img2.src = '/angels-logos.jpg';
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (password === 'actuate2026') {
      setAuthenticated(true);
      sessionStorage.setItem('angels-auth', '1');
    } else {
      setPassword('');
    }
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

  if (!authenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <form onSubmit={handleSubmit} className="flex flex-col items-center gap-4">
          <h1
            className="text-5xl md:text-7xl font-season text-navy mb-8"
            style={{ fontVariationSettings: '"wght" 500, "SERF" 70', lineHeight: '0.9', letterSpacing: '-0.01em' }}
          >
            Dream Machines
          </h1>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="Password"
            autoFocus
            className="border border-navy/20 px-3 py-2 text-base lg:text-lg font-dm-mono text-navy bg-transparent focus:outline-none focus:ring-2 focus:ring-navy/30 placeholder:text-navy-muted w-64"
          />
          <button
            type="submit"
            className="px-5 py-2.5 text-base lg:text-lg bg-navy text-cream font-dm-mono font-medium hover:bg-navy-light transition-colors"
          >
            Enter
          </button>
        </form>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative">
      {/* Content */}
      <div className="max-w-[1088px] mx-auto px-6 flex flex-col items-center">
        {/* Hero Heading */}
        <div className="text-center mt-20">
          <h1
            className="text-5xl md:text-7xl lg:text-[110px] font-season text-navy"
            style={{ fontVariationSettings: '"wght" 500, "SERF" 70', lineHeight: '0.9', letterSpacing: '-0.01em' }}
          >
            Dream Machines
          </h1>
        </div>

        {/* Body text */}
        <div className="max-w-[758px] 2xl:max-w-[620px] mt-[70px] text-base lg:text-lg leading-[1.3] font-dm-mono text-navy space-y-6">
          <p>
            Robotic AI will equalise the cost of labour across the globe. This is the biggest economic shift of our lifetime, and it will be led by whoever figures out how to deploy robots not in billion-dollar factories, but in the small and mid-sized manufacturers where most of the world&apos;s products are actually made.
          </p>

          <p>
            Dream Machines is building robotic AI that allows anyone who understands a product or process to automate it. Starting with single tasks on a workbench, growing to teams of robots working together, and ultimately enabling non-technical people to set up entire production lines without a single human working in them. The logical starting point is the smallest unit of value: one task, one arm, one worker teaching it what to do.
          </p>

          <p>
            <strong>Small and mid-sized manufacturers have no viable automation path today.</strong><span className="block h-[0.35em]" />
            Today, when a manufacturer wants to automate a single workbench task, they face two options. Hire a systems integrator at 5 to 20 times the cost of the robot arm itself, wait months to find out if it works, and end up with a solution so rigid that any change to the product or the workstation means calling them back.
          </p>

          <p>
            Or just pay someone to do it by hand. Most choose the latter, but increasingly, they cannot find the people either. For every automatable task still being done manually today, there is a multiple of work not being done at all because European factories cannot find the people they need. Entire production lines have already moved to Asia, taking decades of process knowledge with them.
          </p>

          <p>
            <strong>Dream Machines lets the people who know the work automate the work.</strong><span className="block h-[0.35em]" />
            An employee from the factory floor demonstrates the task to a bimanual robot arm. A model trains overnight. The robot starts working the next morning and improves over time. The worker grows from operator to robot supervisor. What used to take months and six figures now takes half a day of demonstrations and a monthly rental at half the cost of an employee. We build the AI pipeline. The arms are off-the-shelf.
          </p>

          <div className="mt-2">
            <Image
              src="/angels-collage.jpg"
              alt="Factory visits, manufacturing work, and cold sales across Europe"
              width={1200}
              height={800}
              className="w-full rounded"
            />
            <p className="text-sm text-navy-muted/60 font-dm-mono mt-2">
              Impressions from 1.5 months on the road doing cold visits. Read more{' '}
              <a href="https://thisiscrispin.substack.com/p/sales-september" target="_blank" rel="noopener noreferrer" className="underline hover:text-navy transition-colors">here</a>.
            </p>
          </div>

          <p>
            <strong>Robotic AI has just become viable, but the path to reliability runs through deployment, not research.</strong><span className="block h-[0.35em]" />
            Robotic AI has crossed a threshold: for the first time, models can learn manipulation tasks from physical demonstration. But &ldquo;works&rdquo; is not &ldquo;works reliably for eight hours on a factory floor.&rdquo; That gap is a data problem. The biggest datasets today are in the low tens of thousands of hours. Getting to general-purpose requires not multiples but orders of magnitude more, and paying for artificial data collection in labs is neither scalable nor cost-efficient. The only source of real, diverse, task-specific data at that scale is deployed robots doing real work for paying customers. Big labs are spending tens of millions brute-forcing the problem with research and chasing general-purpose directly. That path runs through single-task deployments, not around them.
          </p>

          <p>
            Which tasks you deploy on matters enormously. And this is exactly the kind of thing you only learn by spending time on factory floors, not in a lab. I spent 1.5 months doing cold visits across central Europe, sleeping in the car between stops, talking to 80+ manufacturers, and working a week in an electronics factory in Płock. At a 99% success rate, a robot doing a 15-second task fails roughly every 17 minutes, essentially requiring constant supervision. But PCB testing involves 15 seconds of handling plus 92 seconds of machine testing where the operator is idle. With the same success rate the robot now runs over 2 hours between resets. The use case does the heavy lifting. We&apos;re starting with device testing, then battery slotting, component sorting, and beyond.
          </p>

          <p>
            <strong>Single-task automation driven by the user is the wedge into full production lines.</strong><span className="block h-[0.35em]" />
            Every deployment generates training data that works in two directions. It makes the same task better and easier to deploy at the next company. And it accumulates across tasks: first within a category like testing, then across manufacturing more broadly, and eventually into models that generalise beyond factories.
          </p>

          <p>
            The people using our system for one task will naturally start seeing others. They know their processes best, and they&apos;ll discover automation applications researchers would never think of: which tasks work today and which ones to revisit after the next model update we ship. It&apos;s the same dynamic playing out in software right now: the most transformative use cases for LLMs weren&apos;t designed by the companies that built them, but discovered by power users who understood their own workflows and are now rebuilding their orgs around them. Dream Machines is building that trajectory for the physical world.
          </p>

          <div className="mt-2">
            <Image
              src="/angels-logos.jpg"
              alt="Logos of companies that sent their products for pilot use cases"
              width={1600}
              height={400}
              className="w-full rounded"
            />
            <p className="text-sm text-navy-muted/60 font-dm-mono mt-2">
              Companies that have sent us their products.
            </p>
          </div>

          <p>
            <strong>Eight companies have sent us their physical products because they want us working on their use case first.</strong><span className="block h-[0.35em]" />
            This is phase 1: three paid deployments, first one starting at the end of May, developing on-site with real tasks and real users. These early deployments where we sit on-site and next to the user let us close the loop between what the model does and what real applications require, optimising the ML pipeline and the user interface before we scale.
          </p>

          <p>
            Phase 2 is a productised self-serve solution by end of year. An SME lifts the arm out of the box, moves it to a workbench, collects an afternoon of demonstrations, and has a working model the next day. No integrator, no custom engineering. Through 2027, we&apos;ll be shipping model and software updates that unlock progressively more complex tasks, from testing and sorting to component assembly.
          </p>

          <p>
            Phase 3 is mobile, collaborative robots that coordinate with each other and allow domain experts to design entire production lines without a single roboticist involved. At this stage, the most interesting market is no longer automating what&apos;s still being done by hand, but enabling new company models where a small team with product knowledge uses our systems to make production fully autonomous.
          </p>

          <p>
            <strong>My background: From business school to NeurIPS published research.</strong><span className="block h-[0.35em]" />
            I studied Math at ETH Zurich, published papers in Nature Communications and NeurIPS, and have been building robotic AI models since early 2025. But unlike 99.9% of the people you&apos;ll find at such a conference I&apos;m the kind of person who&apos;ll sleep in a car for 1.5 months, do cold walk-ins with 80+ manufacturers and work in a Polish electronics factory to understand what SME automation means and how buyers and users think. It&apos;s this combination of technical depth and go-to-market intensity that Dream Machines is built on and the culture we&apos;re hiring for.
          </p>

          <p>
            I&apos;m now raising an angel round to hire three engineers and finance hardware for the first deployments.
          </p>
        </div>

        {/* Contact note */}
        <div className="max-w-[758px] 2xl:max-w-[620px] w-full mt-8 mb-16">
          <p className="text-navy-muted font-dm-mono text-base lg:text-lg leading-[1.3]">
            <a href="mailto:dominique@dream-machines.eu" className="underline hover:text-navy transition-colors">dominique@dream-machines.eu</a>
          </p>
        </div>
      </div>
    </div>
  );
}
