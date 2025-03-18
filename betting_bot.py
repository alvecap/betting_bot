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
import pytz  # Pour gérer les fuseaux horaires
import os   # Pour les variables d'environnement
import random  # Pour sélectionner des matchs supplémentaires si nécessaire

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
    MIN_PREDICTIONS: int = 5  # Nombre minimum de prédictions à collecter

@dataclass
class Match:
    home_team: str
    away_team: str
    competition: str
    region: str
    commence_time: datetime
    bookmakers: List[Dict]
    all_odds: List[Dict]
    priority: int = 0
    predicted_score1: str = ""
    predicted_score2: str = ""
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0

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
    home_odds: float = 0.0
    draw_odds: float = 0.0
    away_odds: float = 0.0

class BettingBot:
    def __init__(self, config: Config):
        self.config = config
        self.bot = telegram.Bot(token=config.TELEGRAM_BOT_TOKEN)
        self.claude_client = anthropic.Anthropic(api_key=config.CLAUDE_API_KEY)
        self.available_predictions = [
            "1X", "X2", "12", 
            "+1.5 buts", "+2.5 buts", "-3.5 buts",
            "1", "2",  # Suppression du match nul comme prédiction
            "-1.5 buts 1ère mi-temps", 
            "+0.5 but 1ère mi-temps", "+0.5 but 2ème mi-temps"
        ]
        self.top_leagues = {
            "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿": 1,
            "Championnat d'Espagne de Football 🇪🇸": 1,
            "Championnat d'Allemagne de Football 🇩🇪": 1,
            "Championnat d'Italie de Football 🇮🇹": 1,
            "Championnat de France de Football 🇫🇷": 1,
            "Ligue des Champions de l'UEFA 🇪🇺": 1,
            "Ligue Europa de l'UEFA 🇪🇺": 1,
            "Championnat de Belgique de Football 🇧🇪": 2,
            "Championnat des Pays-Bas de Football 🇳🇱": 2,
            "Championnat du Portugal de Football 🇵🇹": 2
        }
        print("Bot initialisé!")

    def _get_league_name(self, competition: str) -> str:
        league_mappings = {
            "Premier League": "Première Ligue Anglaise 🏴󠁧󠁢󠁥󠁮󠁧󠁿",
            "La Liga": "Championnat d'Espagne de Football 🇪🇸",
            "Bundesliga": "Championnat d'Allemagne de Football 🇩🇪",
            "Serie A": "Championnat d'Italie de Football 🇮🇹",
            "Ligue 1": "Championnat de France de Football 🇫🇷",
            "Champions League": "Ligue des Champions de l'UEFA 🇪🇺",
            "Europa League": "Ligue Europa de l'UEFA 🇪🇺",
            "Belgian First Division A": "Championnat de Belgique de Football 🇧🇪",
            "Dutch Eredivisie": "Championnat des Pays-Bas de Football 🇳🇱",
            "Primeira Liga": "Championnat du Portugal de Football 🇵🇹"
        }
        return league_mappings.get(competition, competition)

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def fetch_matches(self, max_match_count: int = 100) -> List[Match]:
        """Récupère jusqu'à 100 matchs du jour"""
        print("\n1️⃣ RÉCUPÉRATION DES MATCHS DU JOUR...")
        url = "https://api.the-odds-api.com/v4/sports/soccer/odds/"
        params = {
            "apiKey": self.config.ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h,totals",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            matches_data = response.json()
            print(f"✅ {len(matches_data)} matchs récupérés depuis l'API")

            current_time = datetime.now(timezone.utc)
            matches = []
            
            # Obtenir la date d'aujourd'hui (sans l'heure) en UTC
            today_utc = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
            tomorrow_utc = today_utc + timedelta(days=1)

            for match_data in matches_data:
                commence_time = datetime.fromisoformat(match_data["commence_time"].replace('Z', '+00:00'))
                
                # Ne prendre que les matchs qui se jouent aujourd'hui
                if today_utc <= commence_time < tomorrow_utc:
                    competition = self._get_league_name(match_data.get("sport_title", "Unknown"))
                    
                    # Extraire les cotes des bookmakers
                    home_odds, draw_odds, away_odds = 0.0, 0.0, 0.0
                    
                    if match_data.get("bookmakers") and len(match_data["bookmakers"]) > 0:
                        for bookmaker in match_data["bookmakers"]:
                            if bookmaker.get("markets") and len(bookmaker["markets"]) > 0:
                                for market in bookmaker["markets"]:
                                    if market["key"] == "h2h" and len(market["outcomes"]) == 3:
                                        for outcome in market["outcomes"]:
                                            if outcome["name"] == match_data["home_team"]:
                                                home_odds = outcome["price"]
                                            elif outcome["name"] == match_data["away_team"]:
                                                away_odds = outcome["price"]
                                            else:
                                                draw_odds = outcome["price"]
                                        break
                            if home_odds > 0 and draw_odds > 0 and away_odds > 0:
                                break
                    
                    # Ne garder que les matchs avec des cotes complètes (home, draw, away)
                    if home_odds > 0 and draw_odds > 0 and away_odds > 0:
                        matches.append(Match(
                            home_team=match_data["home_team"],
                            away_team=match_data["away_team"],
                            competition=competition,
                            region=competition.split()[-1] if " " in competition else competition,
                            commence_time=commence_time,
                            bookmakers=match_data.get("bookmakers", []),
                            all_odds=match_data.get("bookmakers", []),
                            priority=self.top_leagues.get(competition, 0),
                            home_odds=home_odds,
                            draw_odds=draw_odds,
                            away_odds=away_odds
                        ))

            if not matches:
                print(f"❌ Aucun match trouvé pour aujourd'hui")
                return []

            # Trier les matchs: d'abord par priorité des ligues, puis par heure de début
            matches.sort(key=lambda x: (-x.priority, x.commence_time))
            
            # Prendre un grand nombre de matchs comme candidats
            top_matches = matches[:max_match_count]
            
            print(f"\n✅ {len(top_matches)} matchs candidats sélectionnés pour aujourd'hui")
            # Afficher les 10 premiers matchs triés pour vérification
            for i, match in enumerate(top_matches[:10]):
                match_time = match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M")
                print(f"{i+1}. {match_time} - {match.home_team} vs {match.away_team} ({match.competition}) - Cotes: {match.home_odds}/{match.draw_odds}/{match.away_odds}")
                
            return top_matches

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des matchs: {str(e)}")
            return []

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def get_match_stats(self, match: Match) -> Optional[str]:
        print(f"\n2️⃣ ANALYSE DE {match.home_team} vs {match.away_team}")
        try:
            # Utiliser le même modèle et prompt que pour les scores exacts
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Tu es une intelligence artificielle experte en analyse sportive de football. 

Fais une analyse détaillée et factuelle pour le match {match.home_team} vs {match.away_team} ({match.competition}) qui aura lieu le {match.commence_time.strftime('%d/%m/%Y')}.

Analyse OBLIGATOIREMENT tous ces éléments:
1. FORME RÉCENTE:
   - 5 derniers matchs de chaque équipe avec les résultats exacts
   - Moyenne de buts marqués/encaissés par match
   - Performance à domicile/extérieur (pourcentage de victoires)

2. CONFRONTATIONS DIRECTES:
   - Les 5 dernières rencontres entre ces équipes avec scores
   - Tendances des confrontations (équipe dominante)
   - Nombre moyen de buts dans ces confrontations

3. STATISTIQUES CLÉS:
   - Pourcentage de matchs avec +1.5 buts pour chaque équipe
   - Pourcentage de matchs avec +2.5 buts pour chaque équipe
   - Pourcentage de matchs avec -3.5 buts pour chaque équipe
   - Pourcentage de matchs où les deux équipes marquent

4. ABSENCES ET EFFECTIF:
   - Joueurs blessés ou suspendus importants
   - Impact des absences sur le jeu de l'équipe

5. CONTEXTE DU MATCH:
   - Enjeu sportif (qualification, maintien, position au classement)
   - Importance du match pour chaque équipe

Sois aussi précis et factuel que possible avec des statistiques réelles."""
                    }],
                    "max_tokens": 700,
                    "temperature": 0.1
                },
                timeout=60  # 1 minute pour obtenir les statistiques
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

    @retry(tries=2, delay=5, backoff=2, logger=logger)
    def get_predicted_scores(self, match: Match) -> Optional[tuple]:
        """Récupère les scores prédits, retourne None si impossible d'obtenir des prédictions fiables"""
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
                        "content": f"""Tu es une intelligence artificielle experte en paris sportifs, spécialisée dans la prédiction de scores exacts. Tu utilises des modèles statistiques avancés, y compris la méthode ELO, pour évaluer la force relative des équipes et estimer le nombre de buts potentiels de chaque équipe dans un match.

Objectif: Générer deux propositions de scores exacts pour le match {match.home_team} vs {match.away_team} qui aura lieu le {match.commence_time.strftime('%d/%m/%Y')} en {match.competition}.

Pour générer ces prédictions, analyse les éléments suivants:
1. Contexte du match (compétition, enjeu, phase du tournoi)
2. Forme et performances des équipes (5 derniers matchs, buts marqués/encaissés)
3. Confrontations directes (historique entre les équipes)
4. Absences et forme des joueurs clés
5. Analyse avec la méthode ELO et statistiques avancées
6. Tendances des bookmakers et experts
7. Facteurs psychologiques et extra-sportifs

Réponds UNIQUEMENT au format "Score 1: X-Y, Score 2: Z-W" où X,Y,Z,W sont des nombres entiers. Ne donne aucune autre information ou explication."""
                    }],
                    "max_tokens": 100,
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
            prompt = f"""ANALYSE APPROFONDIE: {match.home_team} vs {match.away_team}
COMPÉTITION: {match.competition}
SCORES EXACTS PRÉDITS: {match.predicted_score1} et {match.predicted_score2}
COTES: Victoire {match.home_team}: {match.home_odds}, Match nul: {match.draw_odds}, Victoire {match.away_team}: {match.away_odds}

DONNÉES STATISTIQUES:
{stats}

CONSIGNES STRICTES:
1. Analyser en profondeur les statistiques fournies et les scores exacts prédits
2. Évaluer les tendances et performances des équipes
3. Considérer les scores exacts prédits par les experts
4. Choisir la prédiction LA PLUS SÛRE parmi: {', '.join(self.available_predictions)}

RÈGLES DE VÉRIFICATION OBLIGATOIRES:
- Pour prédire une victoire à domicile "1", l'équipe à domicile doit avoir une cote MAXIMALE de 1.50
- Pour prédire une victoire à l'extérieur "2", l'équipe extérieure doit avoir une cote MAXIMALE de 1.50
- Si la cote est supérieure à 1.50, NE PAS prédire de victoire directe; préférer double chance (1X ou X2)
- Pour prédire "+1.5 buts", on doit être sûr à 90% que le match aura AU MOINS 3 BUTS
- Pour prédire "+2.5 buts", on doit être sûr à 90% que le match aura AU MOINS 4 BUTS
- Pour prédire "-3.5 buts", la probabilité doit être d'au moins 80% que le match aura moins de 4 buts
- Ne jamais donner de prédiction sans une confiance d'au moins 80%
- Le match nul "X" n'est PAS une option de prédiction acceptable
- Privilégier les prédictions avec les statistiques les plus solides

FORMAT REQUIS:
PREDICTION: [une option de la liste]
CONFIANCE: [pourcentage]"""

            message = self.claude_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )

            response = message.content[0].text
            prediction = re.search(r"PREDICTION:\s*(.*)", response)
            confidence = re.search(r"CONFIANCE:\s*(\d+)", response)

            if all([prediction, confidence]):
                pred = prediction.group(1).strip()
                conf = min(100, max(80, int(confidence.group(1))))
                
                if any(p.lower() in pred.lower() for p in self.available_predictions):
                    # Trouver la prédiction exacte dans la liste
                    for available_pred in self.available_predictions:
                        if available_pred.lower() in pred.lower():
                            pred = available_pred
                            break
                    
                    # Vérifications supplémentaires pour la fiabilité
                    # Appliquer les règles strictes pour les victoires directes
                    if pred == "1" and match.home_odds > 1.50:
                        print(f"⚠️ Cote domicile trop élevée ({match.home_odds} > 1.50). Conversion en 1X.")
                        pred = "1X"
                    elif pred == "2" and match.away_odds > 1.50:
                        print(f"⚠️ Cote extérieur trop élevée ({match.away_odds} > 1.50). Conversion en X2.")
                        pred = "X2"
                    
                    if pred == "X":
                        print("⚠️ Prédiction de match nul non autorisée. Prédiction rejetée.")
                        return None
                    
                    print(f"✅ Prédiction finale: {pred} (Confiance: {conf}%)")
                    
                    return Prediction(
                        region=match.region,
                        competition=match.competition,
                        match=f"{match.home_team} vs {match.away_team}",
                        time=match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M"),
                        predicted_score1=match.predicted_score1,
                        predicted_score2=match.predicted_score2,
                        prediction=pred,
                        confidence=conf,
                        home_odds=match.home_odds,
                        draw_odds=match.draw_odds,
                        away_odds=match.away_odds
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
            # Formatage des éléments avec gras et italique - SANS LES COTES
            msg += (
                f"*📊 MATCH #{i}*\n"
                f"🏆 _{pred.competition}_\n"
                f"*⚔️ {pred.match}*\n"
                f"⏰ Coup d'envoi : _{pred.time}_\n"
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
            
            # 1. Récupérer jusqu'à 100 matchs du jour (tri par priorité déjà fait dans fetch_matches)
            all_matches = self.fetch_matches(max_match_count=100)
            
            if not all_matches:
                print("❌ Aucun match trouvé pour aujourd'hui")
                return
                
            if len(all_matches) < self.config.MAX_MATCHES:
                print(f"⚠️ Attention: Seulement {len(all_matches)} matchs disponibles pour aujourd'hui")
                print("⚠️ Le bot tentera d'obtenir autant de prédictions que possible")
            
            # 2. Initialiser pour le traitement
            predictions = []
            processed_matches = set()
            attempts = 0
            max_attempts = min(len(all_matches) * 2, 100)  # Limiter le nombre de tentatives
            
            # 3. Continuer jusqu'à obtenir exactement 5 prédictions ou avoir épuisé toutes les possibilités
            while len(predictions) < self.config.MAX_MATCHES and attempts < max_attempts and len(processed_matches) < len(all_matches):
                # Trouver les matchs non traités
                available_indices = [i for i in range(len(all_matches)) if i not in processed_matches]
                if not available_indices:
                    break
                
                # Prendre le prochain match disponible
                match_index = available_indices[0]  # Prendre le premier match disponible (déjà trié par priorité)
                current_match = all_matches[match_index]
                processed_matches.add(match_index)
                
                attempts += 1
                match_time = current_match.commence_time.astimezone(timezone(timedelta(hours=1))).strftime("%H:%M")
                print(f"\n⏳ Tentative {attempts}/{max_attempts}: Match {len(processed_matches)}/{len(all_matches)}: {current_match.home_team} vs {current_match.away_team} ({match_time})")
                
                # ÉTAPE 1: Obtenir les scores prédits
                scores = self.get_predicted_scores(current_match)
                if not scores:
                    print(f"⚠️ Échec à l'étape des scores prédits. Passage au match suivant.")
                    continue
                    
                current_match.predicted_score1, current_match.predicted_score2 = scores
                
                # ÉTAPE 2: Obtenir les statistiques
                stats = self.get_match_stats(current_match)
                if not stats:
                    print(f"⚠️ Échec à l'étape des statistiques. Passage au match suivant.")
                    continue
                
                # ÉTAPE 3: Analyser le match pour obtenir une prédiction
                prediction = self.analyze_match(current_match, stats)
                if not prediction:
                    print(f"⚠️ Échec à l'étape de l'analyse. Passage au match suivant.")
                    continue
                
                # ÉTAPE 4: Ajouter la prédiction à notre liste
                predictions.append(prediction)
                print(f"✅ Prédiction #{len(predictions)}/{self.config.MAX_MATCHES} obtenue")
                
                # Pause entre les matchs pour éviter de surcharger les API
                if len(predictions) < self.config.MAX_MATCHES and len(processed_matches) < len(all_matches):
                    await asyncio.sleep(3)
            
            # 4. Résumé final des prédictions
            print(f"\n📊 {len(processed_matches)}/{len(all_matches)} matchs traités, {len(predictions)}/{self.config.MAX_MATCHES} prédictions obtenues")
            
            # 5. Vérifier si nous avons obtenu suffisamment de prédictions
            if len(predictions) >= self.config.MAX_MATCHES:
                print(f"✅ Nombre requis de prédictions atteint: {len(predictions)}/{self.config.MAX_MATCHES}")
                
                # Si nous avons trop de prédictions, ne prendre que les 5 meilleures (par confiance)
                if len(predictions) > self.config.MAX_MATCHES:
                    predictions.sort(key=lambda p: p.confidence, reverse=True)
                    predictions = predictions[:self.config.MAX_MATCHES]
                
                # Envoyer les prédictions
                await self.send_predictions(predictions)
                print("=== ✅ EXÉCUTION TERMINÉE AVEC SUCCÈS ===")
            
            # Si nous avons des prédictions mais pas assez
            elif 0 < len(predictions) < self.config.MAX_MATCHES:
                print(f"⚠️ Seulement {len(predictions)}/{self.config.MAX_MATCHES} prédictions obtenues après {attempts} tentatives")
                print("ℹ️ Envoi du coupon avec moins de 5 prédictions plutôt que rien")
                await self.send_predictions(predictions)
                print("=== ⚠️ EXÉCUTION TERMINÉE AVEC AVERTISSEMENT ===")
            
            # Si nous n'avons obtenu aucune prédiction
            else:
                print("❌ Aucune prédiction fiable n'a pu être générée après épuisement des matchs disponibles")
                print("=== ❌ EXÉCUTION TERMINÉE AVEC ÉCHEC ===")

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

async def scheduler():
    print("Démarrage du bot de paris sportifs...")
    
    # Configuration à partir des variables d'environnement
    config = Config(
        TELEGRAM_BOT_TOKEN=os.environ.get("TELEGRAM_BOT_TOKEN", "votre_token_telegram"),
        TELEGRAM_CHAT_ID=os.environ.get("TELEGRAM_CHAT_ID", "votre_chat_id"),
        ODDS_API_KEY=os.environ.get("ODDS_API_KEY", "votre_cle_odds"),
        PERPLEXITY_API_KEY=os.environ.get("PERPLEXITY_API_KEY", "votre_cle_perplexity"),
        CLAUDE_API_KEY=os.environ.get("CLAUDE_API_KEY", "votre_cle_claude"),
        MAX_MATCHES=int(os.environ.get("MAX_MATCHES", "5")),
        MIN_PREDICTIONS=int(os.environ.get("MIN_PREDICTIONS", "5"))
    )
    
    bot = BettingBot(config)
    
    # Vérifier si l'exécution immédiate est demandée
    RUN_ON_STARTUP = os.environ.get("RUN_ON_STARTUP", "true").lower() == "true"
    
    # Envoyer un message de test au démarrage
    await send_test_message(bot.bot, config.TELEGRAM_CHAT_ID)
    
    # Exécuter immédiatement si RUN_ON_STARTUP est vrai
    if RUN_ON_STARTUP:
        print("Exécution immédiate au démarrage...")
        await bot.run()
    
  # Boucle principale du scheduler
    while True:
        # Heure actuelle en Afrique centrale (UTC+1)
        africa_central_time = pytz.timezone("Africa/Lagos")  # Lagos est en UTC+1
        now = datetime.now(africa_central_time)
        
        # Exécution planifiée à 7h00
        if now.hour == 7 and now.minute == 0:
            print(f"Exécution planifiée du bot à {now.strftime('%Y-%m-%d %H:%M:%S')}")
            await bot.run()
            
            # Attendre 1 minute pour éviter les exécutions multiples
            await asyncio.sleep(60)
        
        # Attendre 1 minute avant de vérifier à nouveau
        await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(scheduler())
