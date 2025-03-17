import asyncio
import requests
import anthropic
import logging
import telegram
import nest_asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import re
import sys
from retry import retry
import pytz  # Pour gérer les fuseaux horaires
import os   # Pour les variables d'environnement
import random  # Pour sélectionner aléatoirement des matchs supplémentaires

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
    MIN_PREDICTIONS: int = 5  # Nombre minimum de prédictions requises

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
            "1", "X", "2",
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
    def fetch_matches(self, max_match_count: int = 15) -> List[Match]:
        """Récupère les matchs depuis l'API avec un nombre plus élevé pour avoir des alternatives"""
        print("\n1️⃣ RÉCUPÉRATION DES MATCHS...")
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
            print(f"✅ {len(matches_data)} matchs récupérés")

            current_time = datetime.now(timezone.utc)
            matches = []

            for match_data in matches_data:
                commence_time = datetime.fromisoformat(match_data["commence_time"].replace('Z', '+00:00'))
                # Prendre les matchs des prochaines 24 heures
                if 0 < (commence_time - current_time).total_seconds() <= 86400:
                    competition = self._get_league_name(match_data.get("sport_title", "Unknown"))
                    matches.append(Match(
                        home_team=match_data["home_team"],
                        away_team=match_data["away_team"],
                        competition=competition,
                        region=competition.split()[-1] if " " in competition else competition,
                        commence_time=commence_time,
                        bookmakers=match_data.get("bookmakers", []),
                        all_odds=match_data.get("bookmakers", []),
                        priority=self.top_leagues.get(competition, 0)
                    ))

            if not matches:
                return []

            # Trier les matchs par priorité et heure de début
            matches.sort(key=lambda x: (-x.priority, x.commence_time))
            
            # Prendre plus de matchs que nécessaire pour avoir des alternatives
            top_matches = matches[:max_match_count]
            
            print(f"\n✅ {len(top_matches)} matchs candidats sélectionnés")
            for match in top_matches[:5]:
                print(f"- {match.home_team} vs {match.away_team} ({match.competition})")
                
            return top_matches

        except Exception as e:
            print(f"❌ Erreur lors de la récupération des matchs: {str(e)}")
            return []

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def get_match_stats(self, match: Match) -> Optional[str]:
        """Récupère les statistiques du match en utilisant le même prompt que pour les scores exacts"""
        print(f"\n2️⃣ ANALYSE DE {match.home_team} vs {match.away_team}")
        try:
            response = requests.post(
                "https://api.perplexity.ai/chat/completions",
                headers={"Authorization": f"Bearer {self.config.PERPLEXITY_API_KEY}",
                        "Content-Type": "application/json"},
                json={
                    "model": "llama-3.1-sonar-large-128k-online",
                    "messages": [{
                        "role": "user", 
                        "content": f"""Tu es une intelligence artificielle experte en analyse sportive, spécialisée dans le football. Tu utilises des modèles statistiques avancés, y compris la méthode ELO, pour analyser les matchs.

Fais une analyse détaillée pour {match.home_team} vs {match.away_team} ({match.competition}) qui aura lieu le {match.commence_time.strftime('%d/%m/%Y')}.

Pour générer cette analyse, tiens compte des éléments suivants:
1. FORME:
   - 5 derniers matchs de chaque équipe (résultats)
   - Buts marqués/encaissés par match récemment
   - Résultats à domicile/extérieur

2. CONFRONTATIONS DIRECTES:
   - Historique des 5 dernières rencontres entre ces équipes
   - Tendances des scores lors de ces confrontations
   - Statistiques de buts dans ces matchs

3. STATISTIQUES IMPORTANTES:
   - Moyenne de buts par match des deux équipes
   - % matchs avec +1.5 buts pour les deux équipes
   - % matchs avec +2.5 buts pour les deux équipes 
   - % matchs avec -3.5 buts pour les deux équipes
   - % victoires/nuls/défaites récents
   - Force à domicile et à l'extérieur

4. EFFECTIF:
   - Blessés et suspendus importants
   - Joueurs clés disponibles pour ce match

5. CONTEXTE DU MATCH:
   - Enjeu sportif (qualification, maintien, titre)
   - Position au classement des deux équipes
   - Série en cours (victoires/défaites consécutives)

Fournis une analyse COMPLÈTE et FACTUELLE qui couvre TOUS les points mentionnés ci-dessus."""
                    }],
                    "max_tokens": 800,
                    "temperature": 0.1
                },
                timeout=180  # Timeout de 3 minutes pour avoir des statistiques complètes
            )
            response.raise_for_status()
            stats = response.json()["choices"][0]["message"]["content"]
            
            # Vérifier que les statistiques sont suffisamment complètes (au moins 300 caractères)
            if len(stats) < 300:
                print("❌ Statistiques obtenues trop limitées, considérées comme incomplètes")
                return None
                
            print("✅ Statistiques réelles récupérées")
            return stats
        except Exception as e:
            print(f"❌ Erreur lors de la récupération des statistiques: {str(e)}")
            return None

    @retry(tries=3, delay=5, backoff=2, logger=logger)
    def get_predicted_scores(self, match: Match) -> Tuple[Optional[str], Optional[str]]:
        """Récupère les scores prédits, retourne None si échec"""
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
                timeout=180  # Timeout de 3 minutes pour obtenir des prédictions fiables
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
                    return None, None
                
        except Exception as e:
            print(f"❌ Erreur lors de l'obtention des scores probables: {str(e)}")
            return None, None

    def analyze_match(self, match: Match, stats: str) -> Optional[Prediction]:
        """Analyse le match avec Claude et retourne une prédiction"""
        print(f"\n4️⃣ ANALYSE AVEC CLAUDE POUR {match.home_team} vs {match.away_team}")
        
        try:
            prompt = f"""ANALYSE APPROFONDIE: {match.home_team} vs {match.away_team}
COMPÉTITION: {match.competition}
SCORES EXACTS PRÉDITS: {match.predicted_score1} et {match.predicted_score2}

DONNÉES STATISTIQUES:
{stats}

CONSIGNES:
1. Analyser en profondeur les statistiques fournies et les scores exacts prédits
2. Évaluer les tendances et performances des équipes
3. Considérer les scores exacts prédits par les experts
4. Choisir la prédiction LA PLUS SÛRE parmi: {', '.join(self.available_predictions)}
5. Justifier avec précision
6. Confiance minimale de 80%

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
                            
                    print(f"✅ Prédiction: {pred} (Confiance: {conf}%)")
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

            print("❌ Pas de prédiction fiable obtenue")
            return None

        except Exception as e:
            print(f"❌ Erreur lors de l'analyse avec Claude: {str(e)}")
            return None

    def _format_predictions_message(self, predictions: List[Prediction]) -> str:
        """Formate le message des prédictions pour Telegram"""
        # Date du jour formatée
        current_date = datetime.now().strftime('%d/%m/%Y')
        
        # En-tête du message avec formatage en gras
        msg = f"*🤖 AL VE AI BOT - PRÉDICTIONS DU {current_date} 🤖*\n\n"

        for i, pred in enumerate(predictions, 1):
            # Formatage des éléments avec gras et italique
            msg += (
                f"*📊 MATCH #{i}*\n"
                f"🏆 _{pred.competition}_\n"
                f"*⚔️ {pred.match}*\n"
                f"⏰ Coup d'envoi : _{pred.time}_\n"
                f"🔮 Scores prédits : *{pred.predicted_score1}* et *{pred.predicted_score2}*\n"
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
        """Envoie les prédictions via Telegram"""
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

    async def process_match(self, match: Match) -> Optional[Prediction]:
        """Traite un match complet et retourne une prédiction si réussie"""
        try:
            # Obtenir les deux scores exacts probables
            score1, score2 = self.get_predicted_scores(match)
            if score1 is None or score2 is None:
                print(f"⚠️ Impossible d'obtenir des scores valides pour {match.home_team} vs {match.away_team}. Match ignoré.")
                return None
                
            match.predicted_score1 = score1
            match.predicted_score2 = score2
            
            # Obtenir les statistiques
            stats = self.get_match_stats(match)
            if not stats:
                print(f"⚠️ Impossible d'obtenir des statistiques pour {match.home_team} vs {match.away_team}. Match ignoré.")
                return None
            
            return self.analyze_match(match, stats)
        except Exception as e:
            print(f"❌ Erreur lors du traitement du match {match.home_team} vs {match.away_team}: {str(e)}")
            return None
            
    async def run(self) -> None:
        """Exécute le processus principal en s'assurant d'avoir des données statistiques réelles"""
        try:
            print(f"\n=== 🤖 AL VE AI BOT - GÉNÉRATION DES PRÉDICTIONS ({datetime.now().strftime('%H:%M')}) ===")
            
            # Récupérer plus de matchs que nécessaire pour avoir des alternatives
            all_matches = self.fetch_matches(max_match_count=15)
            if not all_matches:
                print("❌ Aucun match trouvé pour aujourd'hui")
                
                # Envoyer un message d'absence de matchs
                await self.bot.send_message(
                    chat_id=self.config.TELEGRAM_CHAT_ID,
                    text="*🤖 AL VE AI BOT - NOTIFICATION*\n\nAucun match prévu dans les prochaines 24 heures.",
                    parse_mode="Markdown"
                )
                return
                
            predictions = []
            processed_matches_count = 0
            
            # Mélanger légèrement les matchs pour varier les prédictions
            # tout en gardant les matchs prioritaires en premier
            priority_matches = all_matches[:self.config.MAX_MATCHES]
            remaining_matches = all_matches[self.config.MAX_MATCHES:]
            random.shuffle(remaining_matches)
            
            all_matches_ordered = priority_matches + remaining_matches
            
            # Traiter les matchs jusqu'à avoir le nombre minimum de prédictions
            for match in all_matches_ordered:
                if len(predictions) >= self.config.MIN_PREDICTIONS:
                    print(f"🎯 Nombre requis de prédictions atteint: {len(predictions)}/{self.config.MIN_PREDICTIONS}")
                    break
                    
                processed_matches_count += 1
                print(f"\n🔍 Analyse du match {processed_matches_count}/{len(all_matches_ordered)}: {match.home_team} vs {match.away_team}")
                
                prediction = await self.process_match(match)
                if prediction:
                    predictions.append(prediction)
                    print(f"✅ Prédiction {len(predictions)}/{self.config.MIN_PREDICTIONS} obtenue")
                else:
                    print(f"⚠️ Pas de prédiction obtenue pour ce match (problème de données)")
                
                # Attendre entre chaque analyse pour ne pas surcharger les API
                await asyncio.sleep(5)
            
            print(f"\n🔄 {processed_matches_count} matchs traités, {len(predictions)} prédictions obtenues")
            
            if len(predictions) >= self.config.MIN_PREDICTIONS:
                print(f"✅ Nombre minimum de prédictions atteint: {len(predictions)}/{self.config.MIN_PREDICTIONS}")
                await self.send_predictions(predictions)
                print("=== ✅ EXÉCUTION TERMINÉE ===")
            else:
                print(f"⚠️ Nombre insuffisant de prédictions: {len(predictions)}/{self.config.MIN_PREDICTIONS}")
                if predictions:
                    # Envoyer quand même les prédictions disponibles
                    await self.send_predictions(predictions)
                    print(f"=== ⚠️ EXÉCUTION TERMINÉE AVEC MOINS DE PRÉDICTIONS QUE REQUIS ===")
                else:
                    # Aucune prédiction disponible
                    await self.bot.send_message(
                        chat_id=self.config.TELEGRAM_CHAT_ID,
                        text="*🤖 AL VE AI BOT - ERREUR*\n\nImpossible de générer des prédictions fiables aujourd'hui. Nouvelle tentative prévue demain.",
                        parse_mode="Markdown"
                    )
                    print("=== ❌ AUCUNE PRÉDICTION GÉNÉRÉE ===")

        except Exception as e:
            print(f"❌ ERREUR GÉNÉRALE: {str(e)}")
            # Notifier de l'erreur
            try:
                await self.bot.send_message(
                    chat_id=self.config.TELEGRAM_CHAT_ID,
                    text=f"*🤖 AL VE AI BOT - ERREUR*\n\nUne erreur s'est produite lors de la génération des prédictions: {str(e)}",
                    parse_mode="Markdown"
                )
            except:
                pass

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
