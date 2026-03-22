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
            Dream Machines is building robotic AI that allows anyone who understands a product or process to automate it, without knowing anything about robotics or AI. Starting with single tasks on a workbench, growing to teams of robots working together, and ultimately enabling non-technical people to set up entire production lines without a single human working in them. The logical starting point is the smallest unit of value: one task, one arm, one worker teaching it what to do.
          </p>

          <p>
            Today, when a manufacturer wants to automate a single workbench task, they face two options. Hire a systems integrator at 5 to 20 times the cost of the robot arm itself, wait months to find out if it works, and end up with a solution so rigid that any change to the product or the workstation means calling them back. Or just pay someone to do it by hand. Most choose the latter, but increasingly they cannot find the people either. For every automatable task still being done manually today, there is a multiple of work not getting done at all because European factories cannot find the people they need. Entire production lines have already moved to Asia as a result.
          </p>

          <p>
            Dream Machines lets the people who know the work automate the work. An employee from the factory floor demonstrates the task to a bimanual robot arm. A model trains overnight. The robot starts working the next morning and improves over time. The worker grows from operator to robot supervisor. What used to take months and six figures now takes half a day of demonstrations and a monthly rental at half the cost of an employee. We build the AI pipeline. The arms are off-the-shelf.
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
            Robotic AI has crossed a threshold: for the first time, models can learn manipulation tasks from physical demonstration. But &ldquo;works&rdquo; is not &ldquo;works reliably for eight hours on a factory floor.&rdquo; That gap is a data problem. The biggest manipulation datasets today are in the low tens of thousands of hours. Getting to general-purpose requires not multiples but orders of magnitude more, and paying for artificial data collection in labs is neither scalable nor cost-efficient. The only source of real, diverse, task-specific data at that scale is deployed robots doing real work for paying customers. Big labs are spending tens of millions chasing general-purpose directly. That path runs through single-task deployments, not around them. Every robot we deploy generates the data that makes the next deployment better, and the one after that possible.
          </p>

          <p>
            Most robotics startups are born out of research and treat go-to-market as an afterthought. We treat it as the core differentiator. I&apos;ve visited 80+ factories and worked assembly lines myself; these environments are nothing like a lab. What you choose to automate matters as much as the model itself.
          </p>

          <p>
            For example: at a 99% success rate, a robot doing a 15-second task fails roughly every 17 minutes, essentially requiring constant supervision. But PCB testing involves 15 seconds of handling plus 92 seconds of machine testing where the operator is idle. Same success rate, but now the robot runs over 2 hours between resets. The use case does the heavy lifting. That insight came from showing up cold and customer conversations, not from a lab. We&apos;re starting with workbench tasks in manufacturing, device testing first, then battery slotting, component sorting, and beyond.
          </p>

          <p>
            Eight companies have sent us their physical products because they want us to work on their use cases first. The first paid deployment is planned for June, with three pilots targeted this summer. By year-end, the goal is a fully productised solution: an SME lifts the arm out of the box, moves it to a workbench, collects an afternoon of demonstrations, and has a working model the next day.
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
            Single-task automation is the wedge. As models improve and our training capabilities evolve &ndash; from post-training today to foundational models built on the task data we&apos;re accumulating &ndash; companies will automate increasingly complex tasks. A manufacturer that started with PCB testing will try part-sorting, then packaging, then single-component assembly, then full-device assembly. Over time, robots begin working alongside each other, coordinated by the same people who know the products and processes. The end state: non-technical domain experts building entire production lines without being roboticists. We&apos;re seeing the same arc in software: from single-point code completions to async agents that let one person manage what used to require an entire team. Dream Machines is that same trajectory for the physical world.
          </p>

          <p>
            I studied Math at ETH Zurich, did NeurIPS-published research, and have been building robotic AI models since early 2025. Yet I originally came from a business school background. Not many people who can build robotic AI models would sleep in a car for 1.5 months, driving through Europe, doing cold visits to 100+ manufacturers, and working in a Polish electronics factory to understand where automation actually breaks down. That combination of technical depth and go-to-market intensity is what Dream Machines is built on.
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
