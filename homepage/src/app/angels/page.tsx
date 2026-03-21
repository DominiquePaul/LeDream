'use client';

import { useState, useEffect } from 'react';
import AsciiEurope from '@/components/AsciiEurope';

export default function Angels() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
    document.documentElement.lang = 'en';
  }, []);

  if (!mounted) {
    return (
      <div className="min-h-screen">
        <div className="max-w-[1088px] mx-auto px-6">
          <div className="h-6 bg-navy/5 rounded animate-pulse mt-[200px]"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative">
      {/* ASCII Globe */}
      <div className="max-w-[1088px] mx-auto px-6">
        <AsciiEurope />
      </div>

      {/* Content centered below globe */}
      <div className="max-w-[1088px] mx-auto px-6 flex flex-col items-center">
        {/* Hero Heading */}
        <div className="text-center mt-10">
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
            Robotic AI will equalise the cost of labour across the globe. This is the biggest economic shift of our lifetime, and it will be led by whoever figures out how to deploy robots not in billion-dollar factories, but in the small and mid-sized manufacturers where most of the world&apos;s products are actually made. Dream Machines is building that.
          </p>

          <p>
            Today, when a 200-person electronics manufacturer wants to automate a single workbench task, they have two options: hire a systems integrator at 5&ndash;20&times; the cost of the robot arm itself, creating a months-long dependency and a solution so rigid that any change requires another project. Or don&apos;t automate at all. Most choose the latter and just pay someone to do the job by hand. Millions of tasks that could be automated never are, because the barrier isn&apos;t the robot but the uncertainty and cost, such that most companies never try.
          </p>

          <p>
            Dream Machines replaces that process. A worker who knows the task demonstrates it to a bimanual robot arm. A model trains overnight. The robot starts working the next morning and improves over time. The worker grows from operator to robot supervisor. What used to take months and six figures now takes half a day of demonstrations and a monthly rental.
          </p>

          <p>
            This is possible now because of a recent technical inflection point in robotic AI introduced by VLMs, video and world models. Foundation models for robotic manipulation, like &pi;0.5/0.6, have made learning from demonstration viable for the first time. But &ldquo;viable&rdquo; is not yet &ldquo;reliable.&rdquo; These models work reasonably well at a variety of tasks in the lab, but they&apos;re not reliably usable for single tasks across long stretches of time in real settings. Closing that reliability gap for real tasks in real factories is our short-term technical focus, and the reason we exist.
          </p>

          <p>
            Data is the biggest bottleneck in robotic AI. Most companies try to solve this by collecting data artificially for use cases they suppose are interesting. This is neither scalable nor capital-efficient. Our approach is structurally different: every paying customer deployment generates real-world task data from day one on tasks that create real value for their business; initially by collecting it themselves and later through the autonomous execution of each workstation.
          </p>

          <p>
            Most robotics startups are born out of research and treat go-to-market as an afterthought. We treat it as the core differentiator, because choosing what to automate matters as much as the model itself. At a 0.99 success rate, a robot doing a 15-second task runs for roughly 17 minutes before failure on median. That requires near-constant supervision. But PCB testing, one of the tasks I identified through one and a half months of cold visits and sleeping in a car, involves 15 seconds of handling plus 92 seconds of machine testing time: 107 seconds of saved human time per cycle. Same success rate, but the robot now runs for over 2 hours before needing intervention. Task selection turns a demo into a deployment.
          </p>

          <p>
            We are starting with workbench tasks in electronics manufacturing: PCB testing, battery slotting, and component sorting. The solution is packaged as a full system (arms, touchscreen interface, training pipeline) and rented by the month. We are planning three paid pilots this summer. By the end of the year, we&apos;re planning a productised solution: an SME lifts the arm out of the box, attaches it to a workbench, and has a working model the next morning.
          </p>

          <p>
            Robotics customers are fundamentally different from software users. SMEs are not tracking benchmarks or swapping providers with a config change. They are trying to do a few things reliably. Once your solution works on their line, they will not risk switching. The relationship is direct, the problem is specific, and churn is near zero. This is why early deployment and customer ownership matter more than papers.
          </p>

          <p>
            The obvious way to size this market is to look at what is currently done by hand. But the real opportunity is larger. When you let people who have problems build their own solutions, the market is not just what is currently being done manually now. It is everything that would be automated if automation did not require a robotics project. A similar analogy is how the really interesting market for Claude Code is not the automation of developers, but people who have problems where the solution is code, but don&apos;t have access to developers.
          </p>

          <p>
            Our product makes automation accessible for non-technical people who know the real-world problems and reduces setup costs to a day. Eight companies have already sent me their physical products because they want to work on their use cases first. The first paid deployment is planned for June.
          </p>

          <p>
            I studied ML and computer vision at ETH Zürich, have NeurIPS-published research, and have been building robotic AI models since early 2025. I originally come from a business background at HSG St. Gallen. Not many people who can build robotic AI models would sleep in a car for 1.5 months, driving through Europe, doing cold visits to 100+ manufacturers, and working in a Polish electronics factory to understand what the current state of automation is and how SME owners think.
          </p>

          <p>
            I&apos;m now raising an angel round to hire 2&ndash;3 engineers and finance hardware for the first deployments.
          </p>
        </div>

        {/* Contact note */}
        <div className="max-w-[758px] 2xl:max-w-[620px] w-full mt-8 mb-16">
          <p className="text-navy-muted font-dm-mono text-base lg:text-lg leading-[1.3]">
            For enquiries, reach out to{' '}
            <a href="mailto:team@dream-machines.eu" className="underline hover:text-navy transition-colors">team@dream-machines.eu</a>
          </p>
        </div>
      </div>
    </div>
  );
}
