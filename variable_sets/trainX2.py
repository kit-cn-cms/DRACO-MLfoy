variables = {}
variables["ge4j_ge3t"] = [
	"Reco_Particle_B1_CSV",
	"Reco_Particle_B2_CSV",
	"Reco_Particle_logM",
	"Reco_Particle_Delta_R",
	# "JetsToParticle_dR",
	# "max_JetsToParticle_dR",
	# "min_JetsToParticle_dR",
	# "NonParticleJets_Pt",
	# "NonParticleJets_E",
	# "Reco_Particle_Pt_ratio",
	]


all_variables = list(set( [v for key in variables for v in variables[key] ] ))
