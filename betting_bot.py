import asyncio
import requests
import anthropic
import logging
import telegram
import nest_asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
import sys
from retry import retry
import pytz
import os
import random
import traceback

# Configuration de base
nest_asyncio.apply()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', 
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

@dataclass
class Config:
    TELEGRAM_BOT_TOKEN: str
    TELEGRAM_CHAT_ID: str
    ODDS_API_KEY: str
    PERPLEXITY_API_KEY: str
    CLAUDE_API_KEY: str
    MAX_MATCHES: int = 5
    MIN_PREDICTIONS: int = 5

@dataclass
class Match:
    home_team: str
    away_team: str
    competition: str
    region: str
    commence_time: datetime
    priority: int = 0
    predicted_score1: str = ""
    predicted_score2: str = ""

@dataclass
class Prediction:
    region: str
    competition: str
    match: str
    time: str
    predicted_score1: str
    predicted_score2: str
    prediction: str
    confidence: int

class BettingBot:
    def __init__(self, config: Config):
        self.config = config
        self.bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
        self.claude_client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        self.available_predictions = [
            "1X", "X2", "12", 
            "+1.5 buts", "+2.5 buts", "-3.5 buts",
            "1", "2",
            "-1.5 buts 1ère mi-temps", 
            "+0.5 but 1ère mi-temps", "+0.5 but 2ème mi-temps"
        ]
        self.top_leagues = {
            # Championnats prioritaires (niveau 1)
            "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 1,
            "Championnat d'Espagne de Football 🇪🇸": 1,
            "Championnat d'Allemagne de Football 🇩🇪": 1,
            "Championnat d'Italie de Football 🇮🇹": 1,
            "Championnat de France de Football 🇫🇷": 1,
            "Ligue des Champions de l'UEFA 🇪🇺": 1,
            "Ligue Europa de l'UEFA 🇪🇺": 1,
            
            # Championnats secondaires (niveau 2)
            "Championnat de Belgique de Football 🇧🇪": 2,
            "Championnat des Pays-Bas de Football 🇳🇱": 2,
            "Championnat du Portugal de Football 🇵🇹": 2,
            "Premier League Russe 🇷🇺": 2,
            "Super League Suisse 🇨🇭": 2,
            "Süper Lig Turque 🇹🇷": 2,
            
            # Compétitions internationales (niveau 1)
            "Coupe du Monde FIFA 🌍": 1,
            "Ligue des Nations UEFA 🇪🇺": 1,
            "Qualifications Coupe du Monde UEFA 🇪🇺": 1,
            "Qualifications Coupe du Monde CAF 🌍": 1,
            "Qualifications Coupe du Monde CONCACAF 🌎": 1,
            "Qualifications Coupe du Monde CONMEBOL 🌎": 1,
            "Qualifications Coupe du Monde AFC 🌏": 1,
            "Qualifications Coupe du Monde OFC 🌏": 1,
            "Coupe d'Afrique des Nations 🌍": 1,
            "Copa America 🌎": 1,
            "Championnat d'Europe UEFA 🇪🇺": 1,
            
            # Autres championnats internationaux (niveau 3)
            "MLS 🇺🇸": 3,
            "Liga MX 🇲🇽": 3,
            "J-League 🇯🇵": 3,
            "K-League 🇰🇷": 3,
            "A-League 🇦🇺": 3,
            "Chinese Super League 🇨🇳": 3,
            "Brasileirão 🇧🇷": 3,
            "Argentine Primera División 🇦🇷": 3
        }
        print("Bot initialisé!")

    def _get_league_name(self, competition: str) -> str:
        league_mappings = {
            # Grands championnats européens
            "Premier League": "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "La Liga": "Championnat d'Espagne de Football 🇪🇸",
            "Bundesliga": "Championnat d'Allemagne de Football 🇩🇪",
            "Serie A": "Championnat d'Italie de Football 🇮🇹",
            "Ligue 1": "Championnat de France de Football 🇫🇷",
            
            # Coupes européennes
            "Champions League": "Ligue des Champions de l'UEFA 🇪🇺",
            "Europa League": "Ligue Europa de l'UEFA 🇪🇺",
            "Conference League": "Ligue Conférence de l'UEFA 🇪🇺",
            
            # Championnats européens secondaires
            "Belgian First Division A": "Championnat de Belgique de Football 🇧🇪",
            "Eredivisie": "Championnat des Pays-Bas de Football 🇳🇱",
            "Primeira Liga": "Championnat du Portugal de Football 🇵🇹",
            "Russian Premier League": "Premier League Russe 🇷🇺",
            "Swiss Super League": "Super League Suisse 🇨🇭",
            "Turkish Super Lig": "Süper Lig Turque 🇹🇷",
            
            # Compétitions internationales
            "FIFA World Cup": "Coupe du Monde FIFA 🌍",
            "UEFA Nations League": "Ligue des Nations UEFA 🇪🇺",
            "UEFA European Championship": "Championnat d'Europe UEFA 🇪🇺",
            "FIFA World Cup Qualification (UEFA)": "Qualifications Coupe du Monde UEFA 🇪🇺",
            "FIFA World Cup Qualification (CAF)": "Qualifications Coupe du Monde CAF 🌍",
            "FIFA World Cup Qualification (CONCACAF)": "Qualifications Coupe du Monde CONCACAF 🌎",
            "FIFA World Cup Qualification (CONMEBOL)": "Qualifications Coupe du Monde CONMEBOL 🌎",
            "FIFA World Cup Qualification (AFC)": "Qualifications Coupe du Monde AFC 🌏",
            "FIFA World Cup Qualification (OFC)": "Qualifications Coupe du Monde OFC 🌏",
            "Africa Cup of Nations": "Coupe d'Afrique des Nations 🌍",
            "Copa America": "Copa America 🌎",
            
            # Autres championnats internationaux
            "Major League Soccer": "MLS 🇺🇸",
            "Liga MX": "Liga MX 🇲🇽",
            "J League": "J-League 🇯🇵",
            "K League 1": "K-League 🇰🇷",
            "A-League": "A-League 🇦🇺",
            "Chinese Super League": "Chinese Super League 🇨🇳",
            "Brasileirão": "Brasileirão 🇧🇷",
            "Argentine Primera División": "Argentine Primera División 🇦🇷"
        }
        return league_mappings.get(competition, competition)

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def fetch_matches(self, max_match_count: int = 30) -> List[Match]:
        """Récupère les matchs sans extraire les cotes"""
        print("\n1️⃣ RÉCUPÉRATION DES MATCHS...")
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"
        params = {
            "apiKey": self.config.ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            matches_data = response.json()
            print(f"✅ {len(matches_data)} matchs récupérés")

            current_time = datetime.now(timezone.utc)
            all_matches = []

            # Collecter tous les matchs sans filtrage complexe
            for match_data in matches_data:
                commence_time = datetime.fromisoformat(match_data["commence_time"].replace('Z', '+00:00'))
                competition = self._get_league_name(match_data.get("sport_title", "Unknown"))
                
                # Prendre tous les matchs des prochaines 48h pour élargir la sélection
                if 0 < (commence_time - current_time).total_seconds() <= 172800:  # 48 heures
                    all_matches.append(Match(
                        home_team=match_data["home_team"],
                        away_team=match_data["away_team"],
                        competition=competition,
                        region=competition.split()[-1] if " " in competition else competition,
                        commence_time=commence_time,
                        priority=self.top_leagues.get(competition, 0)
                    ))

            if not all_matches:
                print("❌ Aucun match trouvé pour les prochaines 48 heures")
                return []

            # Sélectionner les matchs en fonction de la priorité et aléatoirement
            all_matches.sort(key=lambda x: (-x.priority, x.commence_time))
            
            # Déterminer combien de matchs sélectionner
            required_matches = self.config.MIN_PREDICTIONS
            
            # S'assurer de sélectionner exactement MIN_PREDICTIONS matchs ou tous si moins disponibles
            if len(all_matches) <= required_matches:
                selected_matches = all_matches
            else:
                # Sélection avec priorité aux championnats majeurs
                high_priority = [m for m in all_matches if m.priority == 1]
                medium_priority = [m for m in all_matches if m.priority == 2]
                other_matches = [m for m in all_matches if m.priority == 0 or m.priority > 2]
                
                selected_matches = []
                
                # Prendre entre 2 et 3 matchs de haute priorité s'ils sont disponibles
                if high_priority:
                    num_high = min(3, len(high_priority))
                    selected_matches.extend(random.sample(high_priority, num_high))
                
                # Prendre 1 ou 2 matchs de priorité moyenne
                if medium_priority and len(selected_matches) < required_matches:
                    num_medium = min(2, len(medium_priority), required_matches - len(selected_matches))
                    if num_medium > 0:
                        selected_matches.extend(random.sample(medium_priority, num_medium))
                
                # Compléter avec d'autres matchs pour atteindre le minimum requis
                remaining_needed = required_matches - len(selected_matches)
                if remaining_needed > 0 and other_matches:
                    num_other = min(remaining_needed, len(other_matches))
                    selected_matches.extend(random.sample(other_matches, num_other))
                
                # Si on n'a toujours pas assez, compléter avec n'importe quels matchs restants
                if len(selected_matches) < required_matches:
                    remaining = [m for m in all_matches if m not in selected_matches]
                    if remaining:
                        still_needed = required_matches - len(selected_matches)
                        selected_matches.extend(random.sample(remaining, min(still_needed, len(remaining))))
            
            print(f"\n✅ {len(selected_matches)} matchs candidats sélectionnés sur {len(all_matches)} matchs disponibles")
            for match in selected_matches:
                print(f"- {match.home_team} vs {match.away_team} ({match.competition})")
                
            return selected_matches

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des matchs: {str(e)}")
            return []

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def get_match_stats(self, match: Match) -> Optional[str]:
        print(f"\n2️⃣ ANALYSE DE {match.home_team} vs {match.away_team}")
        try:
            # Prompt amélioré pour des statistiques plus complètes
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Tu es une intelligence artificielle experte en analyse sportive de football avec accès aux données les plus récentes. 

Fais une analyse DÉTAILLÉE et FACTUELLE pour le match {match.home_team} vs {match.away_team} ({match.competition}) qui aura lieu le {match.commence_time.strftime('%d/%m/%Y')}.

Analyse OBLIGATOIREMENT et avec PRÉCISION tous ces éléments:

1. FORME RÉCENTE (DÉTAILLÉE):
   - 5 derniers matchs de chaque équipe avec scores exacts, dates et contexte
   - Moyenne précise de buts marqués/encaissés par match (domicile/extérieur)
   - Performance à domicile/extérieur (pourcentage de victoires, nuls, défaites)
   - Tendances récentes des deux équipes (forme ascendante/descendante)
   - Buts marqués/encaissés par mi-temps

2. CONFRONTATIONS DIRECTES (HISTORIQUE COMPLET):
   - Les 5 dernières rencontres entre ces équipes avec scores, dates et contexte
   - Tendances des confrontations (équipe dominante, nombre de buts)
   - Nombre moyen de buts dans ces confrontations
   - Résultats à domicile/extérieur dans les confrontations directes

3. STATISTIQUES CLÉS (PRÉCISES):
   - Pourcentage exact de matchs avec +1.5 buts pour chaque équipe
   - Pourcentage exact de matchs avec +2.5 buts pour chaque équipe
   - Pourcentage exact de matchs avec -3.5 buts pour chaque équipe
   - Pourcentage exact de matchs où les deux équipes marquent
   - Statistiques de possession et d'occasions créées

4. ABSENCES ET EFFECTIF (DÉTAILS COMPLETS):
   - Liste des joueurs blessés ou suspendus importants
   - Impact des absences sur le jeu de l'équipe
   - Joueurs clés disponibles et leur influence
   - État de forme des buteurs principaux

5. CONTEXTE DU MATCH (ANALYSE COMPLÈTE):
   - Enjeu sportif (qualification, maintien, position au classement)
   - Importance du match pour chaque équipe
   - Contexte mental et dynamique d'équipe
   - Facteurs externes (météo prévue, état du terrain)
   - Tactiques probables des entraîneurs

Sois absolument précis et factuel avec des statistiques réelles et vérifiables."""
                    }],
                    "max_tokens": 850,  # Augmenté pour obtenir plus de détails
                    "temperature": 0.1
                },
                timeout=90  # Augmenté pour permettre une analyse plus approfondie
            )
            response.raise_for_status()
            stats = response.json()["choices"][0]["message"]["content"]
            print("✅ Statistiques complètes récupérées")
            return stats
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des statistiques: {str(e)}")
            
            # En cas d'échec, essayer avec un prompt plus court
            try:
                print("⚠️ Tentative avec un prompt simplifié...")
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                            "Content-Type": "application/json"},
                    json={
                        "model": "sonar",
                        "messages": [{
                            "role": "user", 
                            "content": f"""Analyse factuelle pour le match {match.home_team} vs {match.away_team} ({match.competition}):

1. Forme récente des deux équipes (résultats des 5 derniers matchs)
2. Confrontations directes récentes
3. Statistiques: matchs avec +1.5 buts, +2.5 buts, -3.5 buts
4. Absences importantes
5. Enjeu du match"""
                        }],
                        "max_tokens": 500,
                        "temperature": 0.1
                    },
                    timeout=45
                )
                response.raise_for_status()
                stats = response.json()["choices"][0]["message"]["content"]
                print("✅ Statistiques basiques récupérées")
                return stats
            except Exception as e:
                print(f"❌ Erreur lors de la récupération des statistiques simplifiées: {str(e)}")
                return None

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def get_predicted_scores(self, match: Match) -> Optional[tuple]:
        """Récupère les scores prédits avec un prompt amélioré pour des prédictions plus précises"""
        print(f"\n3️⃣ OBTENTION DES SCORES EXACTS PROBABLES POUR {match.home_team} vs {match.away_team}")
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Tu es une intelligence artificielle experte en analyse de football et prédiction de scores exacts, avec accès aux données les plus récentes du monde du football. Tu utilises un système avancé de modélisation statistique qui intègre:

1. MÉTHODE ELO AVANCÉE: 
   - Calcul précis de la force relative de chaque équipe
   - Prise en compte des performances récentes (pondération exponentielle)
   - Ajustement pour l'avantage du terrain

2. MODÈLE DE POISSON:
   - Distribution de Poisson pour estimer la probabilité de chaque nombre de buts
   - Analyse des moyennes de buts marqués/encaissés récentes

3. ANALYSE CONTEXTUELLE COMPLÈTE:
   - Importance du match (enjeu sportif, classement, qualification, derby)
   - Phase de la saison (début, milieu, fin, période chargée)
   - Contexte de la compétition (championnat, coupe, Europe)
   - Série en cours (victoires consécutives, défaites, nuls)

4. ANALYSE DES EFFECTIFS:
   - Joueurs clés absents (blessés, suspendus)
   - Alignements probables et impact des changements
   - Retour de joueurs importants
   - Fatigue (rotation, calendrier chargé)

5. FACTEURS EXTERNES:
   - Conditions météorologiques prévues (pluie, chaleur, vent)
   - État du terrain
   - Déplacements (distance, décalage horaire)
   - Affluence et atmosphère (domicile/extérieur/neutre)

6. STYLE DE JEU ET TACTIQUES:
   - Compatibilité des styles de jeu
   - Approche tactique probable des entraîneurs
   - Adaptation tactique selon la forme récente

7. TENDANCES HISTORIQUES:
   - Confrontations directes récentes (5 derniers matchs)
   - Performance des équipes en fonction du contexte similaire
   - Tendances de buts par période (1ère/2ème mi-temps)

OBJECTIF: Générer DEUX propositions de scores exacts pour le match {match.home_team} vs {match.away_team} ({match.competition}) prévu le {match.commence_time.strftime('%d/%m/%Y')}.

Ces scores doivent refléter l'issue la plus probable du match selon ton analyse complète. Utilise TOUTES les données mentionnées ci-dessus pour une prédiction précise.

RÉPONDS UNIQUEMENT AU FORMAT EXACT: "Score 1: X-Y, Score 2: Z-W" où X, Y, Z et W sont des nombres entiers. Ne donne AUCUNE autre information ou explication."""
                    }],
                    "max_tokens": 150,
                    "temperature": 0.1
                },
                timeout=120  # 2 minutes pour les scores exacts
            )
            response.raise_for_status()
            prediction_text = response.json()["choices"][0]["message"]["content"].strip()
            
            # Extraire les deux scores
            score1_match = re.search(r'Score 1:\s*(\d+)-(\d+)', prediction_text)
            score2_match = re.search(r'Score 2:\s*(\d+)-(\d+)', prediction_text)
            
            if score1_match and score2_match:
                score1 = f"{score1_match.group(1)}-{score1_match.group(2)}"
                score2 = f"{score2_match.group(1)}-{score2_match.group(2)}"
                print(f"✅ Scores probables obtenus: {score1} et {score2}")
                return score1, score2
            else:
                # Si le format n'est pas respecté, extraire les scores si possible
                scores = re.findall(r'(\d+)[^\d]+(\d+)', prediction_text)
                if len(scores) >= 2:
                    score1 = f"{scores[0][0]}-{scores[0][1]}"
                    score2 = f"{scores[1][0]}-{scores[1][1]}"
                    print(f"✅ Scores probables extraits: {score1} et {score2}")
                    return score1, score2
                else:
                    print("❌ Format de scores invalide, match ignoré")
                    return None
                
        except Exception as e:
            print(f"❌ Erreur lors de l'obtention des scores probables: {str(e)}")
            return None

    def analyze_match(self, match: Match, stats: str) -> Optional[Prediction]:
        print(f"\n4️⃣ ANALYSE AVEC CLAUDE POUR {match.home_team} vs {match.away_team}")
        
        try:
            # Prompt amélioré pour une analyse plus fine et des prédictions plus fiables
            prompt = f"""ANALYSE APPROFONDIE POUR PRÉDICTION DE PARIS: {match.home_team} vs {match.away_team}
COMPÉTITION: {match.competition}
SCORES EXACTS PRÉDITS: {match.predicted_score1} et {match.predicted_score2}

DONNÉES STATISTIQUES COMPLÈTES:
{stats}

CONSIGNES D'ANALYSE AVANCÉE:
1. Analyser MÉTICULEUSEMENT les statistiques fournies et les scores exacts prédits
2. Évaluer les tendances historiques et performances récentes avec PRÉCISION
3. Considérer le CONTEXTE COMPLET du match (enjeu, classement, motivation)
4. Analyser l'IMPACT des absences et retours sur l'équilibre des forces
5. Prendre en compte les FACTEURS EXTERNES (météo, terrain, déplacement)
6. Évaluer la COMPATIBILITÉ DES STYLES de jeu des deux équipes
7. Considérer la FIABILITÉ HISTORIQUE des équipes pour maintenir un résultat
8. Choisir la prédiction LA PLUS SÛRE possible parmi: {', '.join(self.available_predictions)}

RÈGLES DE VÉRIFICATION STRICTES:
- Pour prédire une victoire à domicile "1", l'équipe à domicile doit avoir une forme récente EXCELLENTE et dominer clairement son adversaire
- Pour prédire une victoire à l'extérieur "2", l'équipe extérieure doit avoir une forme récente EXCELLENTE et être nettement supérieure
- Pour les victoires directes (1 ou 2), exiger une confiance d'au moins 90% pour cette prédiction
- Pour prédire "1X", l'équipe à domicile doit être favorite selon les statistiques ET les scores prédits
- Pour prédire "X2", l'équipe à l'extérieur doit être favorite selon les statistiques ET les scores prédits
- Ne JAMAIS donner "X2" si les scores prédits favorisent l'équipe à domicile
- Ne JAMAIS donner "1X" si les scores prédits favorisent l'équipe à l'extérieur
- Pour prédire "+1.5 buts", il faut être sûr à 90% que le match aura AU MOINS 2 BUTS
- Pour prédire "+2.5 buts", il faut être sûr à 90% que le match aura AU MOINS 3 BUTS
- Pour prédire "-3.5 buts", la probabilité doit être d'au moins 85% que le match aura MOINS DE 4 BUTS
- Exiger une confiance d'au moins 85% pour TOUTE prédiction
- Le match nul "X" n'est PAS une option de prédiction acceptable
- Privilégier les prédictions avec les statistiques les plus SOLIDES et COHÉRENTES
- En cas de doute, préférer une prédiction concernant le nombre de buts plutôt qu'une double chance ou résultat direct non justifié

FORMAT DE RÉPONSE REQUIS:
PREDICTION: [une option UNIQUE de la liste]
CONFIANCE: [pourcentage précis entre 80 et 100]
JUSTIFICATION: [explication brève de la prédiction]"""

            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.1,  # Réduit pour plus de cohérence
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text
            prediction = re.search(r"PREDICTION:\s*(.*)", response)
            confidence = re.search(r"CONFIANCE:\s*(\d+)", response)
            justification = re.search(r"JUSTIFICATION:\s*(.*?)($|\n\n)", response, re.DOTALL)

            if all([prediction, confidence]):
                pred = prediction.group(1).strip()
                conf = min(100, max(80, int(confidence.group(1))))
                reason = justification.group(1).strip() if justification else ""
                
                if any(p.lower() in pred.lower() for p in self.available_predictions):
                    # Trouver la prédiction exacte dans la liste
                    for available_pred in self.available_predictions:
                        if available_pred.lower() in pred.lower():
                            pred = available_pred
                            break
                    
                    # Pour les victoires directes, vérification stricte de la confiance
                    if pred in ["1", "2"] and conf < 90:
                        print(f"⚠️ Confiance insuffisante ({conf}% < 90%) pour victoire directe. Conversion en double chance.")
                        pred = "1X" if pred == "1" else "X2"
                    
                    # 2. Vérifier la cohérence des doubles chances avec les scores prédits
                    if pred == "X2":
                        # Extraire les scores pour vérifier si l'équipe extérieure est réellement favorisée
                        home_goals1, away_goals1 = map(int, match.predicted_score1.split('-'))
                        home_goals2, away_goals2 = map(int, match.predicted_score2.split('-'))
                        
                        # Si les deux scores prédits favorisent l'équipe à domicile, rejeter X2
                        if home_goals1 > away_goals1 and home_goals2 > away_goals2:
                            print(f"⚠️ Incohérence: X2 prédit mais les scores {match.predicted_score1} et {match.predicted_score2} favorisent l'équipe à domicile.")
                            # Proposer une prédiction alternative sur les buts
                            total_goals1 = home_goals1 + away_goals1
                            total_goals2 = home_goals2 + away_goals2
                            if total_goals1 >= 3 or total_goals2 >= 3:
                                pred = "+2.5 buts"
                                print(f"✅ Prédiction ajustée à {pred} pour cohérence avec les scores prédits")
                            else:
                                print("❌ Impossible de trouver une prédiction cohérente. Match ignoré.")
                                return None
                    
                    elif pred == "1X":
                        # Extraire les scores pour vérifier si l'équipe à domicile est réellement favorisée
                        home_goals1, away_goals1 = map(int, match.predicted_score1.split('-'))
                        home_goals2, away_goals2 = map(int, match.predicted_score2.split('-'))
                        
                        # Si les deux scores prédits favorisent l'équipe à l'extérieur, rejeter 1X
                        if home_goals1 < away_goals1 and home_goals2 < away_goals2:
                            print(f"⚠️ Incohérence: 1X prédit mais les scores {match.predicted_score1} et {match.predicted_score2} favorisent l'équipe à l'extérieur.")
                            # Proposer une prédiction alternative sur les buts
                            total_goals1 = home_goals1 + away_goals1
                            total_goals2 = home_goals2 + away_goals2
                            if total_goals1 >= 3 or total_goals2 >= 3:
                                pred = "+2.5 buts"
                                print(f"✅ Prédiction ajustée à {pred} pour cohérence avec les scores prédits")
                            elif total_goals1 >= 2 or total_goals2 >= 2:
                                pred = "+1.5 buts"
                                print(f"✅ Prédiction ajustée à {pred} pour cohérence avec les scores prédits")
                            else:
                                print("❌ Impossible de trouver une prédiction cohérente. Match ignoré.")
                                return None
                    
                    if pred == "X":
                        print("⚠️ Prédiction de match nul non autorisée. Prédiction rejetée.")
                        return None
                    
                    print(f"✅ Prédiction finale: {pred} (Confiance: {conf}%)")
                    if reason:
                        print(f"   Justification: {reason}")
                    
                    return Prediction(
                        region=match.region,
                        competition=match.competition,
                        match=f"{match.home_team} vs {match.away_team}",
                        time=match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M"),
                        predicted_score1=match.predicted_score1,
                        predicted_score2=match.predicted_score2,
                        prediction=pred,
                        confidence=conf
                    )

            print("❌ Pas de prédiction fiable")
            return None

        except Exception as e:
            print(f"❌ Erreur lors de l'analyse avec Claude: {str(e)}")
            return None

    def _format_predictions_message(self, predictions: List[Prediction]) -> str:
        # Date du jour formatée
        current_date = datetime.now().strftime('%d/%m/%Y')
        
        # En-tête du message avec formatage en gras
        msg = f"*🤖 AL VE AI BOT - PRÉDICTIONS DU {current_date} 🤖*\n\n"

        for i, pred in enumerate(predictions, 1):
            # Formatage des éléments avec gras et italique
            msg += (
                f"*📊 MATCH #{i}*\n"
                f"🏆 *{pred.competition}*\n"
                f"*⚔️ {pred.match}*\n"
                f"⏰ Coup d'envoi : *{pred.time}*\n"
                f"🔮 Scores prédits : *{pred.predicted_score1}* ou *{pred.predicted_score2}*\n"
                f"📈 Prédiction : *{pred.prediction}*\n"
                f"✅ Confiance : *{pred.confidence}%*\n\n"
                f"{'─' * 20}\n\n"
            )

        # Pied de page avec formatage en gras et italique
        msg += (
            "*⚠️ RAPPEL IMPORTANT :*\n"
            "• _Pariez de manière responsable_\n"
            "• _Ne dépassez pas 5% de votre bankroll_\n"
            "• *Ces prédictions sont basées sur l'analyse de données*"
        )
        return msg

    async def send_predictions(self, predictions: List[Prediction]) -> None:
        if not predictions:
            print("❌ Aucune prédiction à envoyer")
            return

        print("\n5️⃣ ENVOI DES PRÉDICTIONS")
        
        try:
            message = self._format_predictions_message(predictions)
            
            # Envoyer un message avec formatage Markdown
            await self.bot.send_message(
                chat_id=self.config.TELEGRAM_CHAT_ID,
                text=message,
                parse_mode="Markdown",  # Activer le formatage Markdown
                disable_web_page_preview=True
            )
            print(f"✅ {len(predictions)} prédictions envoyées!")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'envoi des prédictions: {str(e)}")

    async def run(self) -> None:
        try:
            print(f"\n=== 🤖 AL VE AI BOT - GÉNÉRATION DES PRÉDICTIONS ({datetime.now().strftime('%H:%M')}) ===")
            all_matches = self.fetch_matches(max_match_count=30)  # Récupérer jusqu'à 30 matchs pour avoir un bon choix
            if not all_matches:
                print("❌ Aucun match trouvé pour aujourd'hui")
                return

            predictions = []
            processed_count = 0
            
            # On continue jusqu'à avoir le nombre minimum requis de prédictions
            # ou jusqu'à épuiser tous les matchs disponibles
            for match in all_matches:
                processed_count += 1
                
                # Si on a atteint le nombre maximum de prédictions, on s'arrête
                if len(predictions) >= self.config.MAX_MATCHES:
                    break
                
                # Obtenir les deux scores exacts probables
                scores = self.get_predicted_scores(match)
                if not scores:
                    print(f"⚠️ Impossible d'obtenir des scores exacts pour {match.home_team} vs {match.away_team}. Match ignoré.")
                    continue
                    
                match.predicted_score1, match.predicted_score2 = scores
                
                # Obtenir les statistiques
                stats = self.get_match_stats(match)
                if not stats:
                    print(f"⚠️ Impossible d'obtenir des statistiques pour {match.home_team} vs {match.away_team}. Match ignoré.")
                    continue
                
                # Analyser le match et obtenir une prédiction
                prediction = self.analyze_match(match, stats)
                if prediction:
                    predictions.append(prediction)
                    print(f"✅ Prédiction {len(predictions)}/{self.config.MAX_MATCHES} obtenue")
                
                # Attendre un peu entre chaque analyse pour ne pas surcharger les API
                await asyncio.sleep(5)  # Attendre 5 secondes entre chaque match
            
            print(f"\n📊 {processed_count} matchs traités, {len(predictions)} prédictions obtenues")
            
            if predictions:
                if len(predictions) >= self.config.MIN_PREDICTIONS:
                    print(f"✅ Nombre requis de prédictions atteint: {len(predictions)}/{self.config.MIN_PREDICTIONS}")
                else:
                    print(f"⚠️ Seulement {len(predictions)}/{self.config.MIN_PREDICTIONS} prédictions obtenues")
                
                # Limiter au nombre maximum de prédictions si nécessaire
                if len(predictions) > self.config.MAX_MATCHES:
                    predictions = predictions[:self.config.MAX_MATCHES]
                
                # Envoyer les prédictions disponibles
                await self.send_predictions(predictions)
                print("=== ✅ EXÉCUTION TERMINÉE ===")
            else:
                print("❌ Aucune prédiction fiable n'a pu être générée")

        except Exception as e:
            print(f"❌ ERREUR GÉNÉRALE: {str(e)}")

async def send_test_message(bot, chat_id):
    """Envoie un message de test pour vérifier la connectivité avec Telegram"""
    try:
        message = "*🤖 AL VE AI BOT - TEST DE CONNEXION*\n\nLe bot de paris a été déployé avec succès et est prêt à générer des prédictions!"
        await bot.send_message(
            chat_id=chat_id,
            text=message,
            parse_mode="Markdown"
        )
        print("✅ Message de test envoyé")
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi du message de test: {str(e)}")

async def run_once():
    """Exécute le bot une seule fois, pour les exécutions via Render cron job"""
    print("Démarrage du bot de paris sportifs en mode exécution unique...")
    
    # Configuration à partir des variables d'environnement
    config = Config(
        TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID", ""),
        ODDS_API_KEY=os.environ.get("ODDS_API_KEY", ""),
        PERPLEXITY_API_KEY=os.environ.get("PERPLEXITY_API_KEY", ""),
        CLAUDE_API_KEY=os.environ.get("CLAUDE_API_KEY", ""),
        MAX_MATCHES=int(os.environ.get("MAX_MATCHES", "5")),
        MIN_PREDICTIONS=int(os.environ.get("MIN_PREDICTIONS", "5"))
    )
    
    bot = BettingBot(config)
    
    # Envoyer un message de test
    await send_test_message(bot.bot, config.TELEGRAM_CHAT_ID)
    
    # Exécuter le bot
    await bot.run()
    
    print("Exécution terminée.")

async def main():
    """Fonction principale qui détermine comment exécuter le bot"""
    try:
        print("Démarrage du bot de paris...")
        
        # Configuration à partir des variables d'environnement
        config = Config(
            TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
            TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID", ""),
            ODDS_API_KEY=os.environ.get("ODDS_API_KEY", ""),
            PERPLEXITY_API_KEY=os.environ.get("PERPLEXITY_API_KEY", ""),
            CLAUDE_API_KEY=os.environ.get("CLAUDE_API_KEY", ""),
            MAX_MATCHES=int(os.environ.get("MAX_MATCHES", "5")),
            MIN_PREDICTIONS=int(os.environ.get("MIN_PREDICTIONS", "5"))
        )
        
        # Initialiser le bot
        bot = BettingBot(config)
        
        # Test de connexion
        await send_test_message(bot.bot, config.TELEGRAM_CHAT_ID)
        
        # Exécution immédiate
        print("⏰ Exécution immédiate au démarrage...")
        await bot.run()
        print("✅ Exécution immédiate terminée")
        
        # Initialiser la date du dernier jour d'exécution à aujourd'hui
        # pour éviter une nouvelle exécution le même jour
        today = datetime.now().day
        
        # Attendre jusqu'à 8h le lendemain
        print("🕒 Passage en mode attente: prochaine exécution planifiée à 8h00...")
        
        # Boucle principale du scheduler
        while True:
            # Heure actuelle en Afrique centrale (UTC+1)
            africa_central_tz = pytz.timezone("Africa/Lagos")  # Lagos est en UTC+1
            now = datetime.now(africa_central_tz)
            
            # Log d'activité toutes les heures (pour vérifier que le scheduler fonctionne)
            if now.minute == 0:
                print(f"Scheduler actif - Heure actuelle: {now.strftime('%Y-%m-%d %H:%M:%S')} (UTC+1)")
            
            # Exécution planifiée à 8h00, uniquement si on est un jour différent d'aujourd'hui
            if now.hour == 8 and now.minute < 10 and now.day != today:
                print(f"⏰ Exécution planifiée du bot à {now.strftime('%Y-%m-%d %H:%M:%S')} (heure d'Afrique centrale)")
                
                # Message de notification de début d'exécution
                await bot.bot.send_message(
                    chat_id=config.TELEGRAM_CHAT_ID,
                    text="*⏰ GÉNÉRATION DES PRÉDICTIONS*\n\nLes prédictions du jour sont en cours de génération, veuillez patienter...",
                    parse_mode="Markdown"
                )
                
                # Exécuter le bot
                await bot.run()
                
                # Mettre à jour la date du jour après l'exécution
                today = now.day
                print(f"✅ Exécution terminée. Prochaine exécution prévue demain à 8h00")
                
                # Attendre un peu après l'exécution pour éviter les doublons
                await asyncio.sleep(600)  # 10 minutes
            
            # Vérifier toutes les 30 secondes
            await asyncio.sleep(30)
    
    except Exception as e:
        print(f"❌ ERREUR CRITIQUE dans la fonction principale: {str(e)}")
        traceback.print_exc()
        
        # En cas d'erreur critique, attendre avant de quitter
        await asyncio.sleep(300)  # 5 minutes

if __name__ == "__main__":
    asyncio.run(main())
